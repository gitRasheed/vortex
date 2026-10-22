// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Expression;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Binds the column names a scan requested to the columns one Vortex file actually has.
 *
 * <p>The scan's schema comes from the user, a catalog, or one inferred file, so a file may spell a column in another
 * case or lack it entirely. Like Spark's Parquet reader, a name matches exactly, or ignoring case when
 * {@code spark.sql.caseSensitive} is false and exactly one file column matches, and a requested column the file does
 * not have reads as null.
 */
final class FileSchemaBinding {
    private final List<Field> fileFields;
    private final boolean caseSensitive;
    private final List<String> presentFileNames = new ArrayList<>();
    private final Map<String, Integer> vectorIndexByName = new HashMap<>();

    FileSchemaBinding(Schema fileSchema, List<String> requestedNames, boolean caseSensitive) {
        this.fileFields = fileSchema.getFields();
        this.caseSensitive = caseSensitive;
        for (String name : requestedNames) {
            Field field = VortexArrowColumnVector.findField(fileFields, Field::getName, name, caseSensitive);
            if (field == null) {
                vectorIndexByName.put(name, -1);
            } else {
                vectorIndexByName.put(name, presentFileNames.size());
                presentFileNames.add(field.getName());
            }
        }
    }

    /** The file's names for the requested columns it has, in request order: the scan's native projection. */
    String[] presentFileNames() {
        return presentFileNames.toArray(new String[0]);
    }

    /**
     * Returns the position of a requested column among the vectors the projected scan returns, or -1 when the file does
     * not have the column.
     */
    int vectorIndex(String requestedName) {
        return vectorIndexByName.getOrDefault(requestedName, -1);
    }

    /**
     * Resolves a predicate's field path to the expression reading it from this file, walking nested struct fields by
     * name, or returns empty when the file does not have the field.
     */
    Optional<Expression> resolve(String[] fieldPath) {
        List<Field> candidates = fileFields;
        Expression expr = Expression.root();
        for (String part : fieldPath) {
            Field field = VortexArrowColumnVector.findField(candidates, Field::getName, part, caseSensitive);
            if (field == null) {
                return Optional.empty();
            }
            expr = Expression.getItem(field.getName(), expr);
            candidates = field.getType() instanceof ArrowType.Struct ? field.getChildren() : List.of();
        }
        return fieldPath.length == 0 ? Optional.empty() : Optional.of(expr);
    }
}
