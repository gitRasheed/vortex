// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splits temporal arrays into components for independent compression.
//!
//! The days part uses `i32`, so second and millisecond timestamps need a range check before the
//! split can be used. Null slots do not restrict the range of an array.

use std::mem::MaybeUninit;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::Scalar;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::timestamp;

/// The `i32` day, second, and subsecond components of a temporal array.
pub struct TemporalParts {
    /// Days since the Unix epoch, carrying the validity of the original timestamps.
    pub days: ArrayRef,
    /// Seconds within the day, with non-nullable storage.
    pub seconds: ArrayRef,
    /// Subseconds within the second, with non-nullable storage.
    pub subseconds: ArrayRef,
}

/// Splits a [`TemporalArray`] into day, second, and subsecond components.
///
/// Splitting the components by granularity creates more small values, which enables better
/// cascading compression.
///
/// # Errors
///
/// Returns an error if [`can_split_temporal`] returns `false` or the storage cannot be executed.
pub fn split_temporal(array: TemporalArray, ctx: &mut ExecutionCtx) -> VortexResult<TemporalParts> {
    let time_unit = array.temporal_metadata().time_unit();
    if matches!(time_unit, TimeUnit::Days) {
        vortex_bail!("Cannot handle day-level data");
    }

    let check_days = match cached_bounds(&array) {
        Some(bounds) if !bounds_fit(&bounds, time_unit) => return Err(days_overflow(time_unit)),
        Some(_) => false,
        None => true,
    };

    let temporal_values = array
        .temporal_values()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    // After this operation, timestamps will be a PrimitiveArray<i64>
    let timestamps = temporal_values
        .clone()
        .into_array()
        .cast(DType::Primitive(
            PType::I64,
            temporal_values.dtype().nullability(),
        ))?
        .execute::<PrimitiveArray>(ctx)?;

    let length = timestamps.len();

    // If we don't [..length], compiler can't infer all 3 or 4 slices are the
    // same length, and zip() in split_slice_* checks iterator boundaries.
    let timestamps = &timestamps.as_slice::<i64>()[..length];

    let mut days: BufferMut<i32> = BufferMut::with_capacity(timestamps.len());
    let mut seconds: BufferMut<i32> = BufferMut::with_capacity(timestamps.len());
    let days_ptr = &mut days.spare_capacity_mut()[..length];
    let seconds_ptr = &mut seconds.spare_capacity_mut()[..length];

    if matches!(time_unit, TimeUnit::Seconds) {
        let all_fit = if check_days {
            split_slice_seconds::<true>(days_ptr, seconds_ptr, timestamps)
        } else {
            split_slice_seconds::<false>(days_ptr, seconds_ptr, timestamps)
        };

        // The loop also checks null slots. Reject the split only if a non-null value is too large.
        if !all_fit && !days_fit(&array, time_unit, ctx)? {
            return Err(days_overflow(time_unit));
        }

        // SAFETY: `split_slice_seconds` initialized all `length` elements of both buffers.
        unsafe {
            days.set_len(length);
            seconds.set_len(length);
        }

        return Ok(TemporalParts {
            days: PrimitiveArray::new(days.freeze(), temporal_values.validity()?).into_array(),
            seconds: seconds.into_array(),
            subseconds: ConstantArray::new(0, length).into_array(),
        });
    }

    let mut subseconds: BufferMut<i32> = BufferMut::with_capacity(timestamps.len());
    let subseconds_ptr = &mut subseconds.spare_capacity_mut()[..length];

    let all_fit = match time_unit {
        TimeUnit::Nanoseconds => {
            split_slice::<1_000_000_000, false>(days_ptr, seconds_ptr, subseconds_ptr, timestamps)
        }
        TimeUnit::Microseconds => {
            split_slice::<1_000_000, false>(days_ptr, seconds_ptr, subseconds_ptr, timestamps)
        }
        TimeUnit::Milliseconds if check_days => {
            split_slice::<1_000, true>(days_ptr, seconds_ptr, subseconds_ptr, timestamps)
        }
        TimeUnit::Milliseconds => {
            split_slice::<1_000, false>(days_ptr, seconds_ptr, subseconds_ptr, timestamps)
        }
        _ => unreachable!("Handled before"),
    };

    // The loop also checks null slots. Reject the split only if a non-null value is too large.
    if !all_fit && !days_fit(&array, time_unit, ctx)? {
        return Err(days_overflow(time_unit));
    }

    // SAFETY: `split_slice` initialized all `length` elements of the three buffers.
    unsafe {
        days.set_len(length);
        seconds.set_len(length);
        subseconds.set_len(length);
    }

    Ok(TemporalParts {
        days: PrimitiveArray::new(days.freeze(), temporal_values.validity()?).into_array(),
        seconds: seconds.into_array(),
        subseconds: subseconds.into_array(),
    })
}

/// Returns whether every non-null timestamp can use an `i32` days part.
///
/// Returns `false` for day units. Empty and all-null arrays fit for every supported unit.
///
/// # Errors
///
/// Returns an error if the storage values cannot be executed or aggregated.
pub fn can_split_temporal(array: &TemporalArray, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    let time_unit = array.temporal_metadata().time_unit();
    if matches!(time_unit, TimeUnit::Days) {
        return Ok(false);
    }

    days_fit(array, time_unit, ctx)
}

fn days_fit(
    array: &TemporalArray,
    time_unit: TimeUnit,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if matches!(time_unit, TimeUnit::Microseconds | TimeUnit::Nanoseconds) {
        return Ok(true);
    }

    let bounds = match cached_bounds(array) {
        Some(bounds) => bounds,
        None => match min_max(
            array.temporal_values(),
            ctx,
            NumericalAggregateOpts::default(),
        )? {
            Some(MinMaxResult { min, max }) => [min, max],
            None => return Ok(true),
        },
    };

    Ok(bounds_fit(&bounds, time_unit))
}

/// Exact min and max cached on the array, which the layout writer computes before compressing.
fn cached_bounds(array: &TemporalArray) -> Option<[Scalar; 2]> {
    let ext = ExtensionArray::from(array);
    let stats = ext.statistics();
    if let (Some(min), Some(max)) = (
        stats.get(Stat::Min).as_exact(),
        stats.get(Stat::Max).as_exact(),
    ) {
        return Some([min, max].map(|bound| bound.as_extension().to_storage_scalar()));
    }

    let stats = array.temporal_values().statistics();
    let min = stats.get(Stat::Min).as_exact()?;
    let max = stats.get(Stat::Max).as_exact()?;

    Some([min, max])
}

fn bounds_fit(bounds: &[Scalar; 2], time_unit: TimeUnit) -> bool {
    bounds.iter().all(|bound| {
        bound
            .as_primitive()
            .as_::<i64>()
            .is_some_and(|ts| timestamp::days_fit(ts, time_unit))
    })
}

fn days_overflow(time_unit: TimeUnit) -> VortexError {
    vortex_err!(
        "{time_unit} timestamps lie more than i32::MAX days from the epoch and cannot be split \
         into days"
    )
}

/// Adds 2^31 to the days and converts to unsigned, placing in-range days below 2^32.
///
/// An out-of-range value sets a bit above bit 31. OR-ing the offsets preserves those bits, so the
/// loops can check the whole slice without a branch for each timestamp.
#[inline]
fn day_offset<const DIVISOR: i64>(timestamp: i64) -> u64 {
    (timestamp / (timestamp::SECONDS_PER_DAY * DIVISOR) - i64::from(i32::MIN)).cast_unsigned()
}

/// Initializes all three output slices and reports whether the days fit `i32`.
///
/// The slices must have the same length. With `CHECK_DAYS` disabled, always returns `true`.
#[inline]
fn split_slice<const DIVISOR: i64, const CHECK_DAYS: bool>(
    days: &mut [MaybeUninit<i32>],
    seconds: &mut [MaybeUninit<i32>],
    subseconds: &mut [MaybeUninit<i32>],
    timestamps: &[i64],
) -> bool {
    let mut day_offsets = 0u64;

    // Computing chunks of 4 elements lets LLVM optimize stores into
    // a 16-byte vector store per chunk.
    let length = timestamps.len();
    let (timestamps, timestamps_rem) = timestamps.as_chunks::<4>();
    let (days, days_rem) = days[..length].as_chunks_mut::<4>();
    let (seconds, seconds_rem) = seconds[..length].as_chunks_mut::<4>();
    let (subseconds, subseconds_rem) = subseconds[..length].as_chunks_mut::<4>();

    let chunks = timestamps.iter().zip(days).zip(seconds).zip(subseconds);
    for (((timestamp, day), second), subsecond) in chunks {
        let mut day_buf = [0i32; 4];
        let mut second_buf = [0i32; 4];
        let mut subsecond_buf = [0i32; 4];
        for k in 0..4 {
            let parts = timestamp::split_with_divisor::<DIVISOR>(timestamp[k]);
            if CHECK_DAYS {
                day_offsets |= day_offset::<DIVISOR>(timestamp[k]);
            }
            day_buf[k] = parts.days;
            second_buf[k] = parts.seconds;
            subsecond_buf[k] = parts.subseconds;
        }
        for k in 0..4 {
            day[k].write(day_buf[k]);
            second[k].write(second_buf[k]);
            subsecond[k].write(subsecond_buf[k]);
        }
    }

    let remainder = timestamps_rem
        .iter()
        .zip(days_rem)
        .zip(seconds_rem)
        .zip(subseconds_rem);
    for (((&ts, day), second), subsecond) in remainder {
        let parts = timestamp::split_with_divisor::<DIVISOR>(ts);
        if CHECK_DAYS {
            day_offsets |= day_offset::<DIVISOR>(ts);
        }
        day.write(parts.days);
        second.write(parts.seconds);
        subsecond.write(parts.subseconds);
    }

    day_offsets >> 32 == 0
}

/// Initializes the days and seconds slices for second timestamps.
///
/// The slices must have the same length. With `CHECK_DAYS` enabled, reports whether all days fit
/// `i32`. Otherwise returns `true`.
#[inline]
fn split_slice_seconds<const CHECK_DAYS: bool>(
    days: &mut [MaybeUninit<i32>],
    seconds: &mut [MaybeUninit<i32>],
    timestamps: &[i64],
) -> bool {
    let mut day_offsets = 0u64;
    for ((day, second), ts) in days.iter_mut().zip(seconds).zip(timestamps) {
        let parts = timestamp::split_with_divisor::<1>(*ts);
        if CHECK_DAYS {
            day_offsets |= day_offset::<1>(*ts);
        }
        day.write(parts.days);
        second.write(parts.seconds);
    }

    day_offsets >> 32 == 0
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_panic;
    use vortex_session::VortexSession;

    use crate::DateTimeParts;
    use crate::TemporalParts;
    use crate::can_split_temporal;
    use crate::split_temporal;
    use crate::timestamp::SECONDS_PER_DAY;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case(Validity::NonNullable)]
    #[case(Validity::AllValid)]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, false, true]))]
    fn test_split_temporal(#[case] validity: Validity) {
        let mut ctx = SESSION.create_execution_ctx();
        let milliseconds = PrimitiveArray::new(
            buffer![
                86_400i64,            // element with only day component
                86_400i64 + 1000,     // element with day + second components
                86_400i64 + 1000 + 1, // element with day + second + sub-second components
            ],
            validity.clone(),
        )
        .into_array();
        let temporal_array =
            TemporalArray::new_timestamp(milliseconds, TimeUnit::Milliseconds, Some("UTC".into()));
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal_array, &mut ctx).unwrap();

        let days_prim = days.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert!(
            days_prim
                .validity()
                .vortex_expect("days validity should be derivable")
                .mask_eq(&validity, days_prim.len(), &mut ctx)
                .unwrap()
        );
        let seconds_prim = seconds.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert!(matches!(
            seconds_prim
                .validity()
                .vortex_expect("seconds validity should be derivable"),
            Validity::NonNullable
        ));
        let subseconds_prim = subseconds.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert!(matches!(
            subseconds_prim
                .validity()
                .vortex_expect("subseconds validity should be derivable"),
            Validity::NonNullable
        ));
    }

    #[rstest]
    #[case::seconds(TimeUnit::Seconds, false)]
    #[case::milliseconds(TimeUnit::Milliseconds, false)]
    #[case::microseconds(TimeUnit::Microseconds, true)]
    #[case::nanoseconds(TimeUnit::Nanoseconds, true)]
    fn test_extreme_timestamps_split_exactly_or_fail(
        #[case] unit: TimeUnit,
        #[case] splittable: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let temporal = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter([
                Some(i64::MAX), // Largest storage value.
                None,           // Null row.
                Some(i64::MIN), // Smallest storage value.
                Some(0),        // Unix epoch.
            ])
            .into_array(),
            unit,
            Some("UTC".into()),
        );
        assert_eq!(can_split_temporal(&temporal, &mut ctx)?, splittable);

        let result = DateTimeParts::try_from_temporal(temporal.clone(), &mut ctx);
        if splittable {
            assert_arrays_eq!(result?, temporal, &mut ctx);
        } else {
            let err = result.unwrap_err();
            assert!(err.to_string().contains("i32::MAX days"), "{err}");
        }

        Ok(())
    }

    #[rstest]
    #[case::seconds(TimeUnit::Seconds, 1)]
    #[case::milliseconds(TimeUnit::Milliseconds, 1_000)]
    fn test_days_at_i32_bounds(
        #[case] unit: TimeUnit,
        #[case] ticks_per_second: i64,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ticks_per_day = SECONDS_PER_DAY * ticks_per_second;
        let max = i64::from(i32::MAX) * ticks_per_day + ticks_per_day - 1;
        let min = i64::from(i32::MIN) * ticks_per_day - ticks_per_day + 1;

        let temporal = |values: Vec<i64>| {
            TemporalArray::new_timestamp(PrimitiveArray::from_iter(values).into_array(), unit, None)
        };
        let fits = temporal(vec![max, 0, min, max, min, 0, max, min, max]);
        let parts = DateTimeParts::try_from_temporal(fits.clone(), &mut ctx)?;
        assert_arrays_eq!(parts, fits, &mut ctx);

        // One past each bound, in the 4-wide chunks and in the remainder.
        for (index, past) in [(1, max + 1), (8, min - 1)] {
            let mut values = vec![0; 9];
            values[index] = past;
            let Err(err) = split_temporal(temporal(values), &mut ctx) else {
                vortex_panic!("timestamp exceeds the i32 days bound by one tick");
            };
            assert!(err.to_string().contains("i32::MAX days"), "{err}");
        }

        Ok(())
    }

    #[test]
    fn test_out_of_range_null_slots_split() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::new(
            buffer![
                0i64,     // Valid epoch.
                i64::MAX, // Null slot outside the days range.
                86_400,   // Valid next day.
                i64::MIN, // Null slot outside the days range.
                1,        // Valid next second.
            ],
            Validity::from_iter([true, false, true, false, true]),
        );
        let temporal = TemporalArray::new_timestamp(values.into_array(), TimeUnit::Seconds, None);
        let parts = DateTimeParts::try_from_temporal(temporal.clone(), &mut ctx)?;
        assert_arrays_eq!(parts, temporal, &mut ctx);

        Ok(())
    }
}
