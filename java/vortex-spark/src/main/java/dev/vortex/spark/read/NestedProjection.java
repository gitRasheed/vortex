// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Prunes the struct fields Spark does not require from the schema a scan reads.
 *
 * <p>Struct fields are pruned at any depth. Array and map columns are read whole: Vortex has no projection over list
 * elements or map values. The read schema records what Spark requires; {@link FileSchemaBinding} builds the native
 * projection that reads it from each file, and decides from the file's own nullability whether a struct can be rebuilt
 * from its pruned fields, since a user or catalog schema can falsely declare a struct NOT NULL.
 */
final class NestedProjection {
    private NestedProjection() {}

    /**
     * The schema the scan reads for {@code requiredSchema}: the table columns Spark requires, in table order, each with
     * the type the scan returns for it.
     */
    static StructType readSchema(StructType tableSchema, StructType requiredSchema) {
        List<StructField> fields = new ArrayList<>();
        for (StructField column : tableSchema.fields()) {
            StructField required = findField(requiredSchema, column.name());
            if (required != null) {
                fields.add(new StructField(
                        column.name(),
                        readType(column.dataType(), required.dataType()),
                        column.nullable(),
                        column.metadata()));
            }
        }
        return new StructType(fields.toArray(new StructField[0]));
    }

    /**
     * The type the scan reads for a value of {@code tableType} when Spark requires only {@code requiredType} of it.
     * Returns {@code tableType} itself when nothing can be pruned.
     */
    static DataType readType(DataType tableType, DataType requiredType) {
        if (!(tableType instanceof StructType tableStruct) || !(requiredType instanceof StructType requiredStruct)) {
            return tableType;
        }
        List<StructField> kept = new ArrayList<>();
        boolean prunedBelow = false;
        for (StructField field : tableStruct.fields()) {
            StructField required = findField(requiredStruct, field.name());
            if (required == null) {
                continue;
            }
            DataType type = readType(field.dataType(), required.dataType());
            prunedBelow |= !type.equals(field.dataType());
            kept.add(new StructField(field.name(), type, field.nullable(), field.metadata()));
        }
        if (kept.size() == tableStruct.size() && !prunedBelow) {
            return tableType;
        }
        return new StructType(kept.toArray(new StructField[0]));
    }

    private static StructField findField(StructType schema, String name) {
        for (StructField field : schema.fields()) {
            if (field.name().equals(name)) {
                return field;
            }
        }
        return null;
    }
}
