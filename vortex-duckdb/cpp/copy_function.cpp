// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "data.hpp"
#include "error.hpp"
#include "vortex_duckdb.h"
#include "table_function.h"
#include "vortex.h"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parsed_data/create_copy_function_info.hpp"

using namespace duckdb;

struct CopyBindData final : TableFunctionData {
    CopyBindData(unique_ptr<CData> ffi_data) : ffi_data(std::move(ffi_data)) {
    }
    unique_ptr<CData> ffi_data;
    // Column names in write order, used to key WRITTEN_FILE_STATISTICS.
    vector<string> column_names;
};

struct CopyGlobalData final : GlobalFunctionData {
    CopyGlobalData(unique_ptr<CData> ffi_data) : ffi_data(std::move(ffi_data)) {
    }

    unique_ptr<CData> ffi_data;
    // Non-owning; set in copy_to_get_written_statistics (before the write) and filled in
    // copy_to_finalize (after the write). Null when the plan does not request statistics.
    CopyFunctionFileStatistics *written_stats = nullptr;
};

unique_ptr<FunctionData> copy_to_bind(ClientContext &,
                                      CopyFunctionBindInput &,
                                      const vector<string> &column_names,
                                      const vector<LogicalType> &column_types) {
    vector<const char *> ffi_column_names(column_names.size());
    for (size_t i = 0; i < column_names.size(); ++i) {
        ffi_column_names[i] = column_names[i].c_str();
    }

    vector<duckdb_logical_type> ffi_column_types(column_types.size());
    for (size_t i = 0; i < column_types.size(); ++i) {
        // duckdb C api doesn't allow passing const LogicalTypes. We never
        // modify input in copy function.
        ffi_column_types[i] =
            reinterpret_cast<duckdb_logical_type>(const_cast<LogicalType *>(&column_types[i]));
    }

    duckdb_vx_error error_out = nullptr;
    const duckdb_vx_data ffi_bind_data = duckdb_copy_function_copy_to_bind(ffi_column_names.data(),
                                                                           ffi_column_names.size(),
                                                                           ffi_column_types.data(),
                                                                           ffi_column_types.size(),
                                                                           &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }
    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_bind_data));
    auto bind = make_uniq<CopyBindData>(std::move(cdata));
    bind->column_names = column_names;
    return bind;
}

unique_ptr<GlobalFunctionData>
copy_to_initialize_global(ClientContext &, FunctionData &bind_data, const string &file_path) {
    void *const ffi_bind = bind_data.Cast<CopyBindData>().ffi_data->DataPtr();

    duckdb_vx_error error_out = nullptr;
    const duckdb_vx_data ffi_global =
        duckdb_copy_function_copy_to_initialize_global(ffi_bind, file_path.c_str(), &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }

    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_global));
    return make_uniq<CopyGlobalData>(std::move(cdata));
}

void copy_to_sink(ExecutionContext &,
                  FunctionData &bind_data,
                  GlobalFunctionData &gstate,
                  LocalFunctionData &,
                  DataChunk &input) {
    void *const ffi_bind = bind_data.Cast<CopyBindData>().ffi_data->DataPtr();
    void *const ffi_global = gstate.Cast<CopyGlobalData>().ffi_data->DataPtr();
    auto ffi_chunk = reinterpret_cast<duckdb_data_chunk>(&input);
    duckdb_vx_error error_out = nullptr;
    duckdb_copy_function_copy_to_sink(ffi_bind, ffi_global, ffi_chunk, &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }
}

// Called before the write (right after global init) when the plan requests
// WRITTEN_FILE_STATISTICS. We stash the target struct and fill it at finalize, once the
// footer statistics exist - mirroring the parquet writer's store-pointer-then-fill pattern.
void copy_to_get_written_statistics(ClientContext &,
                                    FunctionData &,
                                    GlobalFunctionData &gstate,
                                    CopyFunctionFileStatistics &statistics) {
    gstate.Cast<CopyGlobalData>().written_stats = &statistics;
}

void copy_to_finalize(ClientContext &, FunctionData &bind_data, GlobalFunctionData &gstate) {
    auto &global = gstate.Cast<CopyGlobalData>();
    void *const ffi_global = global.ffi_data->DataPtr();
    duckdb_vx_error error_out = nullptr;
    duckdb_copy_function_copy_to_finalize(ffi_global, &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }

    if (!global.written_stats) {
        return;
    }
    // Fill the statistics captured in copy_to_get_written_statistics from the finished write's
    // footer. min/max come back as owned duckdb values; DuckLake expects their string form.
    auto &names = bind_data.Cast<CopyBindData>().column_names;
    duckdb_vx_written_file_statistics file_stats;
    if (!duckdb_copy_function_get_written_file_statistics(ffi_global, &file_stats)) {
        // Statistics were requested (written_stats is set) but the finished write produced none;
        // that is an internal inconsistency, not a silently empty result.
        throw InternalException("vortex COPY: written statistics were requested but not produced");
    }
    global.written_stats->row_count = file_stats.row_count;
    global.written_stats->file_size_bytes = file_stats.file_size_bytes;
    global.written_stats->footer_size_bytes = Value::UBIGINT(file_stats.footer_size_bytes);
    // Keyed by top-level column name only. The vortex footer reports one statistics set per
    // top-level field, so nested struct/list leaf columns get no statistics here (unlike parquet,
    // which recurses to leaf paths). Flat tables are fully covered.
    for (idx_t i = 0; i < file_stats.num_columns && i < names.size(); i++) {
        duckdb_vx_written_column_statistics col_stats {};
        duckdb_vx_error col_error = nullptr;
        if (!duckdb_copy_function_get_written_column_statistics(ffi_global, i, &col_stats, &col_error)) {
            if (col_error) {
                throw ExecutorException(IntoErrString(col_error));
            }
            // No statistics for this column (e.g. a type without min/max); skip it.
            continue;
        }
        case_insensitive_map_t<Value> column;
        column["num_values"] = Value::UBIGINT(col_stats.num_values);
        if (col_stats.has_column_size) {
            column["column_size_bytes"] = Value::UBIGINT(col_stats.column_size_bytes);
        }
        if (col_stats.has_null_count) {
            column["null_count"] = Value::UBIGINT(col_stats.null_count);
        }
        if (col_stats.min) {
            column["min"] = Value(reinterpret_cast<Value *>(col_stats.min)->ToString());
            duckdb_destroy_value(&col_stats.min);
        }
        if (col_stats.max) {
            column["max"] = Value(reinterpret_cast<Value *>(col_stats.max)->ToString());
            duckdb_destroy_value(&col_stats.max);
        }
        if (col_stats.has_nan_stat) {
            column["has_nan"] = Value::BOOLEAN(col_stats.contains_nan);
        }
        // DuckLake keys column statistics by a quoted, dot-separated path (see
        // DuckLakeUtil::ParseQuotedList); match the parquet writer, which quotes each name.
        global.written_stats->column_statistics.emplace(KeywordHelper::WriteQuoted(names[i], '"'),
                                                        std::move(column));
    }
}

extern "C" duckdb_state duckdb_vx_register_copy_function(duckdb_database ffi_db) {
    D_ASSERT(ffi_db);
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;

    CopyFunction fn("vortex");
    fn.copy_to_bind = copy_to_bind;
    fn.copy_to_initialize_global = copy_to_initialize_global;
    fn.copy_to_initialize_local = [](auto &, auto &) {
        return make_uniq<LocalFunctionData>();
    };
    fn.copy_to_sink = copy_to_sink;
    fn.copy_to_finalize = copy_to_finalize;
    fn.copy_to_get_written_statistics = copy_to_get_written_statistics;
    fn.extension = "vortex";

    // TODO(joe): expose this via c our api
    fn.execution_mode = [](bool, bool) {
        return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
    };
    // TODO(joe): handle parameters as in table_function

    try {
        Catalog &system_catalog = Catalog::GetSystemCatalog(db);
        CatalogTransaction data = CatalogTransaction::GetSystemTransaction(db);
        CreateCopyFunctionInfo copy_info(std::move(fn));
        system_catalog.CreateCopyFunction(data, copy_info);
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to create Vortex copy function:\t" + data.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}
