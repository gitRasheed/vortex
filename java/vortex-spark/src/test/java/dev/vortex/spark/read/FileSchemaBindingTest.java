// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.util.List;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests the native projection {@link FileSchemaBinding} builds for a file, through the shape it then expects back. */
final class FileSchemaBindingTest {
    private static final Field X = new Field("x", FieldType.nullable(new ArrowType.Int(64, true)), List.of());
    private static final Field Y = new Field("y", FieldType.nullable(new ArrowType.Int(64, true)), List.of());
    private static final StructType INNER_X =
            new StructType().add("inner", new StructType().add("x", DataTypes.LongType));

    @Test
    @DisplayName("A struct is rebuilt from its pruned fields only when the file stores it non-nullable")
    void prunesBelowAStructByTheFilesNullability() {
        Field inner = struct("inner", true, X, Y);
        StructType read = new StructType().add("s", INNER_X);

        // pack drops a struct's own nulls, so a nullable struct is selected and its kept fields are read whole.
        FileSchemaBinding nullable = new FileSchemaBinding(new Schema(List.of(struct("s", true, inner))), read, false);
        assertDoesNotThrow(() -> nullable.checkShape(new Schema(List.of(struct("s", true, inner)))));

        FileSchemaBinding nonNullable =
                new FileSchemaBinding(new Schema(List.of(struct("s", false, inner))), read, false);
        assertDoesNotThrow(
                () -> nonNullable.checkShape(new Schema(List.of(struct("s", false, struct("inner", true, X))))));
    }

    @Test
    @DisplayName("A nullable struct below a rebuilt one is selected, so the struct it keeps is read whole")
    void selectsANullableStructBelowARebuiltOne() {
        Schema file = new Schema(List.of(struct("s", false, struct("inner", true, struct("t", false, X, Y)))));
        StructType tX = new StructType().add("t", new StructType().add("x", DataTypes.LongType));
        StructType read = new StructType().add("s", new StructType().add("inner", tX));

        FileSchemaBinding binding = new FileSchemaBinding(file, read, false);
        // s is rebuilt with pack; inner is selected to keep its validity, which reads t whole rather than packing x.
        assertDoesNotThrow(() -> binding.checkShape(file));
    }

    @Test
    @DisplayName("A scan returning other fields than the projection asked for is rejected")
    void checkShapeRejectsOtherFields() {
        Schema whole = new Schema(List.of(struct("s", true, X, Y)));
        StructType pruned = new StructType().add("s", new StructType().add("y", DataTypes.LongType));

        FileSchemaBinding binding = new FileSchemaBinding(whole, pruned, false);
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> binding.checkShape(whole));
        assertEquals("The Vortex scan returned columns [s[x, y]] but the projection asked for [s[y]]", e.getMessage());
    }

    private static Field struct(String name, boolean nullable, Field... children) {
        FieldType type = nullable
                ? FieldType.nullable(ArrowType.Struct.INSTANCE)
                : FieldType.notNullable(ArrowType.Struct.INSTANCE);
        return new Field(name, type, List.of(children));
    }
}
