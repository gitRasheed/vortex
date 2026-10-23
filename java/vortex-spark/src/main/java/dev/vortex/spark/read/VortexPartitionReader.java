// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DataSource;
import dev.vortex.api.Expression;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.spark.VortexFilePartition;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Per-{@link VortexFilePartition} columnar reader.
 *
 * <p>Opens a single Vortex {@link Session}, {@link DataSource} and {@link Scan} spanning all of
 * {@link VortexFilePartition#paths()} and streams every Vortex partition's record batches through the
 * {@link PartitionReader} interface.
 */
final class VortexPartitionReader implements PartitionReader<ColumnarBatch> {
    private final VortexFilePartition spark;
    private final BufferAllocator allocator;
    private final boolean caseSensitive;

    // Held so the DataSource/Scan stay reachable even if the JVM-wide singleton is
    // ever reset during a task; the actual native session is owned by
    // {@link VortexSparkSession} and is not released when this reader closes.
    private Session session;
    private DataSource dataSource;
    private FileSchemaBinding binding;
    private Scan scan;

    private Partition currentPartition;
    private ArrowReader currentReader;
    private boolean currentBatchLoaded;
    private boolean exhausted;

    VortexPartitionReader(
            VortexFilePartition spark,
            List<String> dataColumnNames,
            Map<String, String> formatOptions,
            Predicate[] pushedPredicates,
            boolean caseSensitive) {
        this.spark = spark;
        this.allocator = ArrowAllocation.rootAllocator();
        this.caseSensitive = caseSensitive;

        session = VortexSparkSession.get(formatOptions);
        dataSource = DataSource.open(session, spark.paths(), formatOptions);

        binding = new FileSchemaBinding(dataSource.arrowSchema(allocator), dataColumnNames, caseSensitive);

        // Always project, even with no columns: a scan without a projection reads every column,
        // whereas an empty select only counts rows, which is all count(*) needs.
        var options =
                ScanOptions.builder().projection(Expression.select(binding.presentFileNames(), Expression.root()));
        if (pushedPredicates != null && pushedPredicates.length > 0) {
            buildFilterExpression(pushedPredicates, binding).ifPresent(options::filter);
        }
        scan = dataSource.scan(options.build());
    }

    private static Optional<Expression> buildFilterExpression(Predicate[] predicates, FileSchemaBinding binding) {
        Expression combined = null;
        for (Predicate predicate : predicates) {
            Optional<Expression> expr = SparkPredicateToVortexExpression.convert(predicate, binding::resolve);
            if (expr.isEmpty()) {
                continue;
            }
            combined = combined == null ? expr.get() : Expression.and(combined, expr.get());
        }
        return Optional.ofNullable(combined);
    }

    @Override
    public boolean next() {
        if (exhausted) {
            return false;
        }
        while (true) {
            if (currentReader != null) {
                try {
                    if (currentReader.loadNextBatch()) {
                        currentBatchLoaded = true;
                        return true;
                    }
                } catch (IOException e) {
                    throw failure("load a batch from", e);
                }
                closeCurrentReader();
            }
            if (!scan.hasNext()) {
                exhausted = true;
                return false;
            }
            currentPartition = scan.next();
            currentReader = currentPartition.scanArrow(allocator);
        }
    }

    @Override
    public ColumnarBatch get() {
        if (!currentBatchLoaded) {
            throw new IllegalStateException("no batch loaded; call next() first");
        }
        currentBatchLoaded = false;

        VectorSchemaRoot root;
        try {
            root = currentReader.getVectorSchemaRoot();
        } catch (IOException e) {
            throw failure("read the loaded batch of", e);
        }

        int rowCount = root.getRowCount();
        List<FieldVector> dataVectors = root.getFieldVectors();
        Map<String, String> partVals = spark.partitionValues();
        StructField[] fields = spark.readSchema().fields();
        if (dataVectors.size() != binding.presentFileNames().length) {
            throw new IllegalStateException(String.format(
                    "Scan of %s returned %d data columns but projected %s for the read schema %s (partition values: %s)",
                    spark.paths(),
                    dataVectors.size(),
                    Arrays.toString(binding.presentFileNames()),
                    spark.readSchema(),
                    partVals));
        }

        // Partition columns are not stored in the files: each is a constant parsed from the file path, and every
        // other column is bound to the vector the projected scan returned for it, or to nulls when the file lacks it.
        ColumnVector[] columns = new ColumnVector[fields.length];
        for (int i = 0; i < fields.length; i++) {
            StructField field = fields[i];
            String partValue = partVals.get(field.name());
            if (partValue != null) {
                columns[i] = PartitionPathUtils.createConstantVector(rowCount, field.dataType(), partValue);
                continue;
            }
            int index = binding.vectorIndex(field.name());
            columns[i] = index < 0
                    ? VortexArrowColumnVector.nulls(field.dataType(), rowCount)
                    : new VortexArrowColumnVector(dataVectors.get(index), field.dataType(), caseSensitive);
        }
        return new ColumnarBatch(columns, rowCount);
    }

    @Override
    public void close() {
        closeCurrentReader();
        // Scan and DataSource native resources are released by VortexCleaner once
        // references are dropped. Session is the JVM-wide singleton and outlives this reader.
        scan = null;
        binding = null;
        dataSource = null;
        session = null;
    }

    /** Wraps a failure with the paths being read, the one fact a stack trace alone does not give. */
    private RuntimeException failure(String what, IOException cause) {
        return new RuntimeException(String.format("Failed to %s %s", what, spark.paths()), cause);
    }

    private void closeCurrentReader() {
        if (currentReader != null) {
            try {
                currentReader.close();
            } catch (IOException e) {
                throw failure("close the reader over", e);
            }
            currentReader = null;
        }
        currentPartition = null;
    }
}
