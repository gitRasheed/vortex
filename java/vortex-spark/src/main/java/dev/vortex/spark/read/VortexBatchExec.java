// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableMap;
import dev.vortex.api.Session;
import dev.vortex.jni.NativeFiles;
import dev.vortex.spark.VortexFilePartition;
import dev.vortex.spark.VortexSparkSession;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.types.StructType;

/** Execution source for batch scans of Vortex file tables. */
public final class VortexBatchExec implements Batch {
    private final List<String> paths;
    private final StructType readSchema;
    private final Map<String, String> formatOptions;
    private final Predicate[] pushedPredicates;
    private List<VortexFilePartition> resolvedPartitions;

    /**
     * Creates a new VortexBatchExec for scanning the specified Vortex files.
     *
     * @param paths the list of file paths to scan
     * @param columns the list of columns to read from the files
     * @param pushedPredicates predicates pushed down by Spark; converted to a single Vortex filter expression at read
     *     time
     */
    public VortexBatchExec(
            List<String> paths, List<Column> columns, Map<String, String> formatOptions, Predicate[] pushedPredicates) {
        this.paths = List.copyOf(paths);
        this.readSchema = CatalogV2Util.v2ColumnsToStructType(columns.toArray(new Column[0]));
        this.formatOptions = Map.copyOf(formatOptions);
        this.pushedPredicates = pushedPredicates == null ? new Predicate[0] : pushedPredicates.clone();
    }

    /**
     * Plans the input partitions for this batch scan.
     *
     * <p>Directory-like entries are expanded to concrete {@code .vortex} files. Each resolved file becomes its own
     * {@link VortexFilePartition}; the partition carries the paths the reader should open, the requested schema, and
     * any Hive-style partition values parsed out of the path below its table root.
     */
    @Override
    public InputPartition[] planInputPartitions() {
        resolvedPartitions = resolvePartitions();
        return resolvedPartitions.toArray(InputPartition[]::new);
    }

    @Override
    public PartitionReaderFactory createReaderFactory() {
        List<VortexFilePartition> files = resolvedPartitions != null ? resolvedPartitions : resolvePartitions();
        Set<String> partitionColumns = files.stream()
                .flatMap(file -> file.partitionValues().keySet().stream())
                .collect(Collectors.toSet());
        List<String> dataColumnNames = Arrays.stream(readSchema.fieldNames())
                .filter(name -> !partitionColumns.contains(name))
                .collect(Collectors.toList());
        return new VortexPartitionReaderFactory(dataColumnNames, formatOptions, pushedPredicates);
    }

    private List<VortexFilePartition> resolvePartitions() {
        Session session = VortexSparkSession.get(formatOptions);
        return paths.stream()
                .flatMap(path -> resolvePartitionValues(session, path, formatOptions).entrySet().stream())
                .map(file -> new VortexFilePartition(
                        List.of(file.getKey()), readSchema, formatOptions, ImmutableMap.copyOf(file.getValue())))
                .collect(Collectors.toList());
    }

    /**
     * Resolves the files of one path (see {@link #resolveVortexPaths}) and parses each file's Hive-style partition
     * values below the path's table root, as {@link PartitionPathUtils#tableRoot} chooses it. Shared with schema
     * inference so that planning and inference agree on the partition columns.
     *
     * @return each file's partition values, keyed by its path in listing order
     */
    public static Map<String, Map<String, String>> resolvePartitionValues(
            Session session, String path, Map<String, String> formatOptions) {
        String tableRoot = PartitionPathUtils.tableRoot(path, formatOptions);
        Map<String, Map<String, String>> values = new LinkedHashMap<>();
        for (String file : resolveVortexPaths(session, List.of(path), formatOptions)) {
            values.put(file, PartitionPathUtils.parsePartitionValues(tableRoot, file));
        }
        return values;
    }

    /**
     * Expands directory-like entries to concrete {@code .vortex} files; entries that already name a {@code .vortex}
     * file are kept as-is. Shared with {@link VortexScan#estimateStatistics()} and schema inference so planning and
     * execution resolve paths identically.
     */
    public static List<String> resolveVortexPaths(
            Session session, List<String> paths, Map<String, String> formatOptions) {
        return paths.stream()
                .flatMap(path -> path.endsWith(".vortex")
                        ? Stream.of(path)
                        : NativeFiles.listFiles(session, path, formatOptions).stream())
                .collect(Collectors.toList());
    }
}
