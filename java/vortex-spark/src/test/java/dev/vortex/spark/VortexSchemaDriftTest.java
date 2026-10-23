// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests reading a directory whose files do not share one schema: the table schema comes from a single file, and the
 * other files are read by column and field name against it.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexSchemaDriftTest {

    private SparkSession spark;

    @TempDir
    private Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexSchemaDriftTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("A file appended with its struct fields in another order is read by field name")
    public void appendedStructFieldOrderReadsByName() {
        String path = tempDir.resolve("reordered_struct").toUri().toString();
        structRows(0, 5, "a", "b", "c")
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(path);
        structRows(5, 10, "c", "a", "b")
                .write()
                .format("vortex")
                .mode(SaveMode.Append)
                .save(path);

        List<Row> expected = new ArrayList<>();
        for (long id = 0; id < 10; id++) {
            expected.add(RowFactory.create(id, id + 100, id + 200, id + 300));
        }
        // Whichever file the schema was inferred from, each row's fields come from the right file column.
        assertEquals(
                expected,
                spark.read()
                        .format("vortex")
                        .load(path)
                        .select("id", "s.a", "s.b", "s.c")
                        .orderBy("id")
                        .collectAsList());
        assertEquals(
                expected.subList(5, 10),
                spark.read()
                        .format("vortex")
                        .load(path)
                        .where("s.b >= 205")
                        .select("id", "s.a", "s.b", "s.c")
                        .orderBy("id")
                        .collectAsList());
    }

    /**
     * Rows {@code id} in {@code [from, to)} with {@code s} holding {@code a = id + 100} and so on, in the given order.
     */
    private Dataset<Row> structRows(long from, long to, String... fieldOrder) {
        org.apache.spark.sql.Column[] fields = Arrays.stream(fieldOrder)
                .map(name -> functions
                        .col("id")
                        .plus(100 * (name.charAt(0) - 'a' + 1))
                        .alias(name))
                .toArray(org.apache.spark.sql.Column[]::new);
        return spark.range(from, to)
                .select(functions.col("id"), functions.struct(fields).alias("s"));
    }

    @Test
    @DisplayName("The schema is inferred from the lexicographically first file, not the first one listed")
    public void schemaComesFromLexicographicallyFirstFile() throws IOException {
        Path first = writeSingleFile("first_source", "first");
        Path other = writeSingleFile("other_source", "other");

        // Many copies of the other file, created before and after the first one in a scrambled order, so that a
        // listing in storage order rarely starts with the lexicographically first file.
        Path dir = tempDir.resolve("drift");
        Files.createDirectories(dir);
        for (String name : List.of("k", "c", "h", "e", "b", "j", "d", "g", "f", "i")) {
            Files.copy(other, dir.resolve("part-" + name + ".vortex"));
            if (name.equals("e")) {
                Files.copy(first, dir.resolve("part-a.vortex"));
            }
        }

        Dataset<Row> table = spark.read().format("vortex").load(dir.toUri().toString());
        assertEquals(List.of("id", "first"), Arrays.asList(table.schema().fieldNames()));
        assertEquals(11 * 3, table.count());
    }

    /** Writes rows {@code id} 0 to 2 with the one extra column {@code column = id} as a single file and returns it. */
    private Path writeSingleFile(String directory, String column) throws IOException {
        Path out = tempDir.resolve(directory);
        spark.range(3)
                .withColumn(column, functions.col("id"))
                .coalesce(1)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(out.toUri().toString());
        try (Stream<Path> files = Files.list(out)) {
            return files.filter(file -> file.toString().endsWith(".vortex"))
                    .findFirst()
                    .orElseThrow();
        }
    }
}
