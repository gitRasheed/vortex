// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link PartitionPathUtils}, which discovers and materializes Hive-style partition columns from file
 * paths.
 *
 * <p>Characterizes partition-value parsing (including URL decoding and the {@code __HIVE_DEFAULT_PARTITION__}
 * sentinel), partition column type inference, and constant-vector materialization for each supported type.
 */
final class PartitionPathUtilsTest {
    // --- parsePartitionValues ---

    @Test
    @DisplayName("Parses key=value segments from a Hive-style partition path")
    void parsesKeyValueSegments() {
        Map<String, String> values = PartitionPathUtils.parsePartitionValues(
                "/data/warehouse", "/data/warehouse/year=2024/month=12/part-0.vortex");
        assertEquals(Map.of("year", "2024", "month", "12"), values);
    }

    @Test
    @DisplayName("Preserves the order of partition segments in the path")
    void preservesSegmentOrder() {
        Map<String, String> values = PartitionPathUtils.parsePartitionValues("/tbl", "/tbl/b=2/a=1/c=3/file.vortex");
        assertEquals(List.of("b", "a", "c"), List.copyOf(values.keySet()));
    }

    @Test
    @DisplayName("Ignores segments without key=value shape")
    void ignoresNonPartitionSegments() {
        Map<String, String> values = PartitionPathUtils.parsePartitionValues("/data", "/data/plain/dir/file.vortex");
        assertTrue(values.isEmpty());
    }

    @Test
    @DisplayName("Ignores segments with an empty key or empty value")
    void ignoresEmptyKeyOrValue() {
        assertTrue(
                PartitionPathUtils.parsePartitionValues("/tbl", "/tbl/=v/file").isEmpty());
        assertTrue(
                PartitionPathUtils.parsePartitionValues("/tbl", "/tbl/k=/file").isEmpty());
    }

    @Test
    @DisplayName("URL-decodes both keys and values")
    void urlDecodesKeysAndValues() {
        Map<String, String> values =
                PartitionPathUtils.parsePartitionValues("/tbl", "/tbl/city=New%20York/file.vortex");
        assertEquals(Map.of("city", "New York"), values);
    }

    @Test
    @DisplayName("Splits on the first '=' so values may contain '='")
    void splitsOnFirstEquals() {
        Map<String, String> values = PartitionPathUtils.parsePartitionValues("/tbl", "/tbl/k=a=b/file.vortex");
        assertEquals(Map.of("k", "a=b"), values);
    }

    @Test
    @DisplayName("Only directories below the table root, and not the file name, are partition levels")
    void parsesOnlyBelowTheTableRoot() {
        assertEquals(
                Map.of("day", "1"),
                PartitionPathUtils.parsePartitionValues("/w/region=eu/t", "file:///w/region=eu/t/day=1/k=v.vortex"));
        assertTrue(PartitionPathUtils.parsePartitionValues("/w/region=eu/t", "/w/region=eu/t/k=v.vortex")
                .isEmpty());
    }

    // --- tableRoot ---

    @Test
    @DisplayName("The table root is the path, the parent of a file path, or the basePath option")
    void tableRootFollowsSparkFileIndex() {
        assertEquals("/t/p=1", PartitionPathUtils.tableRoot("/t/p=1", Map.of()));
        assertEquals("/t/p=1", PartitionPathUtils.tableRoot("/t/p=1/a.vortex", Map.of()));
        assertEquals("/t", PartitionPathUtils.tableRoot("/t/p=1/a.vortex", Map.of("basePath", "/t")));
        assertEquals("file:///t", PartitionPathUtils.tableRoot("/t/p=1", Map.of("BASEPATH", "file:///t")));
        assertThrows(
                IllegalArgumentException.class, () -> PartitionPathUtils.tableRoot("/t/p=1", Map.of("basePath", "/u")));
    }

    // --- inferPartitionColumnType ---

    @Test
    @DisplayName("Infers IntegerType for values that fit in an int")
    void infersInteger() {
        assertEquals(DataTypes.IntegerType, PartitionPathUtils.inferPartitionColumnType("42"));
        assertEquals(DataTypes.IntegerType, PartitionPathUtils.inferPartitionColumnType("-7"));
    }

    @Test
    @DisplayName("Infers LongType for integral values wider than an int")
    void infersLong() {
        assertEquals(DataTypes.LongType, PartitionPathUtils.inferPartitionColumnType("9999999999"));
    }

    @Test
    @DisplayName("Infers DoubleType for decimal-point values")
    void infersDouble() {
        assertEquals(DataTypes.DoubleType, PartitionPathUtils.inferPartitionColumnType("3.14"));
    }

    @Test
    @DisplayName("Infers BooleanType for true/false in any case")
    void infersBoolean() {
        assertEquals(DataTypes.BooleanType, PartitionPathUtils.inferPartitionColumnType("true"));
        assertEquals(DataTypes.BooleanType, PartitionPathUtils.inferPartitionColumnType("FALSE"));
    }

    @Test
    @DisplayName("Falls back to StringType for everything else")
    void fallsBackToString() {
        assertEquals(DataTypes.StringType, PartitionPathUtils.inferPartitionColumnType("hello"));
        assertEquals(DataTypes.StringType, PartitionPathUtils.inferPartitionColumnType("2024-12-01"));
    }

    @Test
    @DisplayName("Null and __HIVE_DEFAULT_PARTITION__ infer StringType")
    void nullAndDefaultPartitionInferString() {
        assertEquals(DataTypes.StringType, PartitionPathUtils.inferPartitionColumnType(null));
        assertEquals(DataTypes.StringType, PartitionPathUtils.inferPartitionColumnType("__HIVE_DEFAULT_PARTITION__"));
    }

    // --- inferCommonPartitionColumnType ---

    @Test
    @DisplayName("Integer values widen to long or double across files")
    void integerWidensToLongOrDouble() {
        assertEquals(DataTypes.LongType, PartitionPathUtils.inferCommonPartitionColumnType(List.of("1", "3000000000")));
        assertEquals(DataTypes.DoubleType, PartitionPathUtils.inferCommonPartitionColumnType(List.of("1", "1.5")));
    }

    @Test
    @DisplayName("Long and double fall back to string rather than lose precision, like any other mix of kinds")
    void mixedKindsFallBackToString() {
        assertEquals(
                DataTypes.StringType,
                PartitionPathUtils.inferCommonPartitionColumnType(List.of("9007199254740993", "1.5")));
        assertEquals(DataTypes.StringType, PartitionPathUtils.inferCommonPartitionColumnType(List.of("true", "1")));
        assertEquals(DataTypes.StringType, PartitionPathUtils.inferCommonPartitionColumnType(List.of("1", "abc")));
    }

    @Test
    @DisplayName("Null values take the type of the other values; a column of only nulls is a string")
    void nullsWidenToTheOtherValues() {
        assertEquals(
                DataTypes.IntegerType,
                PartitionPathUtils.inferCommonPartitionColumnType(
                        Arrays.asList(null, "__HIVE_DEFAULT_PARTITION__", "7")));
        assertEquals(
                DataTypes.StringType,
                PartitionPathUtils.inferCommonPartitionColumnType(Arrays.asList(null, "__HIVE_DEFAULT_PARTITION__")));
    }

    // --- createConstantVector ---

    @Test
    @DisplayName("Null and __HIVE_DEFAULT_PARTITION__ materialize as null vectors")
    void nullValuesMaterializeAsNull() {
        ConstantColumnVector fromNull = PartitionPathUtils.createConstantVector(3, DataTypes.StringType, null);
        assertTrue(fromNull.isNullAt(0));

        ConstantColumnVector fromSentinel =
                PartitionPathUtils.createConstantVector(3, DataTypes.StringType, "__HIVE_DEFAULT_PARTITION__");
        assertTrue(fromSentinel.isNullAt(0));
    }

    @Test
    @DisplayName("String values materialize as UTF8 strings")
    void stringMaterializes() {
        ConstantColumnVector vec = PartitionPathUtils.createConstantVector(2, DataTypes.StringType, "us-east");
        assertFalse(vec.isNullAt(0));
        assertEquals(UTF8String.fromString("us-east"), vec.getUTF8String(0));
    }

    @Test
    @DisplayName("Integral values materialize with the width of the target type")
    void integralTypesMaterialize() {
        assertEquals(
                42,
                PartitionPathUtils.createConstantVector(1, DataTypes.IntegerType, "42")
                        .getInt(0));
        assertEquals(
                9999999999L,
                PartitionPathUtils.createConstantVector(1, DataTypes.LongType, "9999999999")
                        .getLong(0));
        assertEquals(
                (short) 7,
                PartitionPathUtils.createConstantVector(1, DataTypes.ShortType, "7")
                        .getShort(0));
        assertEquals(
                (byte) 3,
                PartitionPathUtils.createConstantVector(1, DataTypes.ByteType, "3")
                        .getByte(0));
    }

    @Test
    @DisplayName("DateType parses the value as days since epoch (int)")
    void dateMaterializesAsInt() {
        assertEquals(
                19000,
                PartitionPathUtils.createConstantVector(1, DataTypes.DateType, "19000")
                        .getInt(0));
    }

    @Test
    @DisplayName("Timestamp types parse the value as micros since epoch (long)")
    void timestampMaterializesAsLong() {
        assertEquals(
                1700000000000000L,
                PartitionPathUtils.createConstantVector(1, DataTypes.TimestampType, "1700000000000000")
                        .getLong(0));
        assertEquals(
                1700000000000000L,
                PartitionPathUtils.createConstantVector(1, DataTypes.TimestampNTZType, "1700000000000000")
                        .getLong(0));
    }

    @Test
    @DisplayName("Boolean, float, and double values materialize with their target types")
    void booleanAndFloatingPointMaterialize() {
        assertTrue(PartitionPathUtils.createConstantVector(1, DataTypes.BooleanType, "true")
                .getBoolean(0));
        assertEquals(
                1.5f,
                PartitionPathUtils.createConstantVector(1, DataTypes.FloatType, "1.5")
                        .getFloat(0));
        assertEquals(
                2.25,
                PartitionPathUtils.createConstantVector(1, DataTypes.DoubleType, "2.25")
                        .getDouble(0));
    }

    @Test
    @DisplayName("Unrecognized target types fall back to UTF8 strings")
    void unrecognizedTypeFallsBackToString() {
        ConstantColumnVector vec =
                PartitionPathUtils.createConstantVector(1, DataTypes.CalendarIntervalType, "whatever");
        assertEquals(UTF8String.fromString("whatever"), vec.getUTF8String(0));
    }
}
