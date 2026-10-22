// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import java.util.Locale;
import java.util.Map;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.SupportsDynamicOverwrite;
import org.apache.spark.sql.connector.write.SupportsTruncate;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.internal.SQLConf;

/**
 * Builder for configuring Vortex write operations.
 *
 * <p>This class is responsible for creating BatchWrite instances that execute the actual write operations to create
 * Vortex files from Spark DataFrames.
 */
public final class VortexWriteBuilder implements WriteBuilder, SupportsTruncate, SupportsDynamicOverwrite {

    /** Spark's write option that overrides {@code spark.sql.sources.partitionOverwriteMode} for one write. */
    private static final String PARTITION_OVERWRITE_MODE_OPTION = "partitionOverwriteMode";

    private final String paths;
    private final LogicalWriteInfo writeInfo;
    private final Map<String, String> options;
    private final Transform[] partitionTransforms;
    private boolean truncate = false;
    private boolean overwriteDynamicPartitions = false;

    /**
     * Creates a new VortexWriteBuilder.
     *
     * @param paths root path for write
     * @param writeInfo logical information about the write operation
     * @param options additional write options
     * @param partitionTransforms partition transforms (may be empty)
     */
    public VortexWriteBuilder(
            String paths, LogicalWriteInfo writeInfo, Map<String, String> options, Transform[] partitionTransforms) {
        this.paths = paths;
        this.writeInfo = writeInfo;
        this.options = options;
        this.partitionTransforms = partitionTransforms;
    }

    /**
     * Builds a Write for executing the write operation.
     *
     * @return a new VortexBatchWrite configured with this builder's settings
     */
    @Override
    public Write build() {
        return new VortexBatchWrite(paths, writeInfo.schema(), options, overwriteMode(), partitionTransforms);
    }

    /**
     * Configures the write operation to truncate existing data.
     *
     * <p>When truncate is enabled, existing Vortex files at the output path will be removed before writing new data,
     * unless Spark's partition overwrite mode is {@code dynamic}, in which case only the partitions written to are
     * replaced.
     *
     * @return this builder for method chaining
     */
    @Override
    public WriteBuilder truncate() {
        this.truncate = true;
        return this;
    }

    /**
     * Configures the write operation to replace only the partitions it writes into, leaving every other partition in
     * place.
     *
     * @return this builder for method chaining
     */
    @Override
    public WriteBuilder overwriteDynamicPartitions() {
        this.overwriteDynamicPartitions = true;
        return this;
    }

    private VortexBatchWrite.OverwriteMode overwriteMode() {
        if (overwriteDynamicPartitions) {
            return VortexBatchWrite.OverwriteMode.DYNAMIC_PARTITIONS;
        }
        if (!truncate) {
            return VortexBatchWrite.OverwriteMode.NONE;
        }
        // DataFrameWriter.save() turns SaveMode.Overwrite into a truncate for every DataSource V2 table, whatever
        // spark.sql.sources.partitionOverwriteMode says, so apply the mode here the way Spark's built-in file
        // sources do: the write option takes precedence over the session configuration.
        String mode = writeInfo
                .options()
                .getOrDefault(
                        PARTITION_OVERWRITE_MODE_OPTION,
                        SQLConf.get().partitionOverwriteMode().toString());
        switch (mode.toUpperCase(Locale.ROOT)) {
            case "DYNAMIC":
                return VortexBatchWrite.OverwriteMode.DYNAMIC_PARTITIONS;
            case "STATIC":
                return VortexBatchWrite.OverwriteMode.TRUNCATE;
            default:
                // Falling back to a truncate would delete the whole table on a typo.
                throw new IllegalArgumentException(String.format(
                        "Invalid value '%s' for write option %s: expected STATIC or DYNAMIC",
                        mode, PARTITION_OVERWRITE_MODE_OPTION));
        }
    }
}
