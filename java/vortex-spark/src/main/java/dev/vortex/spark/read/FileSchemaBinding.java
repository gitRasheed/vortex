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
import java.util.function.UnaryOperator;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Binds the schema a scan reads to the columns one Vortex file actually has, and builds the native projection that
 * reads it.
 *
 * <p>The scan's schema comes from the user, a catalog, or one inferred file, so a file may spell a column in another
 * case or lack it entirely. Like Spark's Parquet reader, a name matches exactly, or ignoring case when
 * {@code spark.sql.caseSensitive} is false and exactly one file column matches, and a requested column the file does
 * not have reads as null.
 *
 * <p>A struct whose read type keeps only some of the file's fields is projected to those fields, by the file's names,
 * with {@code select}, which keeps the struct's validity but reads the kept fields whole. A struct the file stores
 * non-nullable is instead rebuilt with {@code pack}, so its kept fields are pruned further. A struct with a declared
 * field the file lacks is read whole: the column vector binds declared fields by name and reads a missing one as null.
 */
final class FileSchemaBinding {
    private final List<Field> fileFields;
    private final boolean caseSensitive;
    private final Map<String, Integer> vectorIndexByName = new HashMap<>();
    private final List<Field> projectedFields = new ArrayList<>();
    private final Expression projection;

    /**
     * A file field's projection, as a function of the expression reading the field whole, the Arrow field the scan
     * returns for it, and whether that is less than the file has. Native expressions are only built when a column is
     * pruned, since each is a native allocation and the plain column selection needs none.
     */
    private record Projected(UnaryOperator<Expression> projection, Field field, boolean pruned) {
        static Projected whole(Field field) {
            return new Projected(UnaryOperator.identity(), field, false);
        }

        /** The expression projecting this field out of {@code parent}, the expression reading the struct holding it. */
        Expression of(Expression parent) {
            return projection.apply(Expression.getItem(field.getName(), parent));
        }
    }

    FileSchemaBinding(Schema fileSchema, StructType readSchema, boolean caseSensitive) {
        this.fileFields = fileSchema.getFields();
        this.caseSensitive = caseSensitive;
        List<Projected> columns = new ArrayList<>();
        boolean pruned = false;
        for (StructField column : readSchema.fields()) {
            Field field = VortexArrowColumnVector.findField(fileFields, Field::getName, column.name(), caseSensitive);
            if (field == null) {
                vectorIndexByName.put(column.name(), -1);
                continue;
            }
            vectorIndexByName.put(column.name(), projectedFields.size());
            Projected projected = project(field, column.dataType());
            pruned |= projected.pruned();
            projectedFields.add(projected.field());
            columns.add(projected);
        }
        String[] names = projectedFields.stream().map(Field::getName).toArray(String[]::new);
        // The plain column selection is kept when no struct is pruned, since the native scan reads it directly.
        this.projection = pruned
                ? Expression.pack(names, pack(columns, Expression.root()), false)
                : Expression.select(names, Expression.root());
    }

    /**
     * Projects the file field {@code file} as the type the read schema declares for it: the field whole, unless it is a
     * struct whose declared fields the file all has and are fewer than the file's or pruned below.
     */
    private Projected project(Field file, DataType declared) {
        if (!(declared instanceof StructType struct) || file.getType().getTypeID() != ArrowType.ArrowTypeID.Struct) {
            return Projected.whole(file);
        }
        List<Field> children = file.getChildren();
        List<Field> kept = new ArrayList<>();
        List<Projected> projected = new ArrayList<>();
        boolean prunedBelow = false;
        for (StructField field : struct.fields()) {
            Field child = VortexArrowColumnVector.findField(children, Field::getName, field.name(), caseSensitive);
            if (child == null) {
                return Projected.whole(file);
            }
            Projected p = project(child, field.dataType());
            prunedBelow |= p.pruned();
            kept.add(child);
            projected.add(p);
        }
        if (kept.size() == children.size() && !prunedBelow) {
            return Projected.whole(file);
        }
        String[] names = kept.stream().map(Field::getName).toArray(String[]::new);
        if (!prunedBelow || file.isNullable()) {
            // Only select keeps a nullable struct's validity, and select reads its fields whole.
            return new Projected(value -> Expression.select(names, value), withChildren(file, kept), true);
        }
        return new Projected(
                value -> Expression.pack(names, pack(projected, value), false),
                withChildren(file, projected.stream().map(Projected::field).toList()),
                true);
    }

    /** The expressions reading {@code fields} out of {@code parent}, each projected as planned. */
    private static Expression[] pack(List<Projected> fields, Expression parent) {
        return fields.stream().map(p -> p.of(parent)).toArray(Expression[]::new);
    }

    private static Field withChildren(Field field, List<Field> children) {
        return new Field(field.getName(), field.getFieldType(), children);
    }

    /** The scan's native projection: the requested columns the file has, by the file's names, with structs pruned. */
    Expression projection() {
        return projection;
    }

    /**
     * Returns the position of a requested column among the vectors the projected scan returns, or -1 when the file does
     * not have the column.
     */
    int vectorIndex(String requestedName) {
        return vectorIndexByName.getOrDefault(requestedName, -1);
    }

    /**
     * Checks that the fields the projected scan returns, including nested struct fields, are the ones the projection
     * asked for, in order. The column vectors bind the read schema's fields to them by name and read a missing field as
     * null, so a projection that returned other fields would otherwise go unnoticed. Only names and nesting are
     * compared; each vector's type is checked against the declared one by
     * {@code VortexArrowColumnVector.checkDeclaredType}.
     *
     * @throws IllegalStateException if the shapes differ
     */
    void checkShape(Schema scanSchema) {
        String expected = shape(projectedFields);
        String actual = shape(scanSchema.getFields());
        if (!expected.equals(actual)) {
            throw new IllegalStateException(String.format(
                    "The Vortex scan returned columns %s but the projection asked for %s", actual, expected));
        }
    }

    private static String shape(List<Field> fields) {
        StringBuilder sb = new StringBuilder("[");
        for (Field field : fields) {
            if (sb.length() > 1) {
                sb.append(", ");
            }
            sb.append(field.getName());
            if (field.getType().getTypeID() == ArrowType.ArrowTypeID.Struct) {
                sb.append(shape(field.getChildren()));
            }
        }
        return sb.append(']').toString();
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
