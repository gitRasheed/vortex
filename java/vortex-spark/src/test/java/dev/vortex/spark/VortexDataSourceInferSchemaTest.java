// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** Tests for schema inference failure reporting in {@link VortexDataSourceV2}. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexDataSourceInferSchemaTest {

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexInferSchemaTest")
                .master("local[1]")
                .config("spark.driver.host", "127.0.0.1")
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
    @DisplayName("Reading a directory without Vortex files reports the offending path")
    public void inferSchemaFailureNamesThePath() {
        String emptyDir = tempDir.toString();

        Throwable thrown = assertThrows(
                Throwable.class, () -> spark.read().format("vortex").load(emptyDir));

        String allMessages = messagesOf(thrown);
        assertTrue(
                allMessages.contains("no .vortex files found"),
                "error should explain that no Vortex files were found, got: " + allMessages);
        assertTrue(allMessages.contains(emptyDir), "error should name the offending path, got: " + allMessages);
    }

    @Test
    @DisplayName("Files inside and outside key=value directories are rejected as a conflicting layout")
    public void mixedPartitionLayoutIsRejected() {
        Path table = tempDir.resolve("mixed_layout");
        Dataset<Row> rows = spark.range(2).toDF("id");
        rows.write().format("vortex").mode(SaveMode.Append).save(table.toString());
        rows.write()
                .format("vortex")
                .mode(SaveMode.Append)
                .save(table.resolve("p=1").toString());

        Throwable thrown = assertThrows(
                Throwable.class,
                () -> spark.read().format("vortex").load(table.toString()).collectAsList());

        String allMessages = messagesOf(thrown);
        assertTrue(
                allMessages.contains("Conflicting partition column names"),
                "error should explain the layout conflict, got: " + allMessages);
        assertTrue(allMessages.contains("p=1"), "error should name an offending path, got: " + allMessages);
    }

    @Test
    @DisplayName("A partition column's type fits the values of every file, whichever file is listed first")
    public void partitionTypeFitsEveryFile() {
        Path table = tempDir.resolve("mixed_types");
        String numeric = table.resolve("p=1").toString();
        String text = table.resolve("p=abc").toString();
        Dataset<Row> rows = spark.range(1).toDF("id");
        rows.write().format("vortex").mode(SaveMode.Append).save(numeric);
        rows.write().format("vortex").mode(SaveMode.Append).save(text);

        for (Dataset<Row> read : List.of(
                spark.read().format("vortex").load(table.toString()),
                spark.read()
                        .format("vortex")
                        .option("basePath", table.toString())
                        .load(numeric, text),
                spark.read()
                        .format("vortex")
                        .option("basePath", table.toString())
                        .load(text, numeric))) {
            assertEquals(DataTypes.StringType, read.schema().apply("p").dataType());
            List<String> values = read.select("p").collectAsList().stream()
                    .map(row -> row.getString(0))
                    .sorted()
                    .collect(Collectors.toList());
            assertEquals(List.of("1", "abc"), values);
        }
    }

    @Test
    @DisplayName("A partition column named like a data column takes the partition type and value")
    public void partitionColumnOverridesDataColumn() {
        Path table = tempDir.resolve("collision");
        Dataset<Row> rows = spark.sql("SELECT 1L AS id, CAST(1.50 AS DECIMAL(10, 2)) AS p");
        rows.write()
                .format("vortex")
                .mode(SaveMode.Append)
                .save(table.resolve("p=7").toString());

        Dataset<Row> read = spark.read().format("vortex").load(table.toString());

        assertEquals(DataTypes.IntegerType, read.schema().apply("p").dataType());
        assertEquals(7, read.select("p").first().getInt(0));
    }

    @Test
    @DisplayName("key=value directories above the table path are not partition columns")
    public void directoriesAboveTheTableAreNotPartitions() {
        Path table = tempDir.resolve("region=eu").resolve("dt=2024-01-01").resolve("table");
        Dataset<Row> rows = spark.sql("SELECT 1L AS id, 'x' AS dt");
        rows.write().format("vortex").mode(SaveMode.Append).save(table.toString());

        Dataset<Row> read = spark.read().format("vortex").load(table.toString());

        assertEquals(List.of("id", "dt"), List.of(read.schema().fieldNames()));
        assertEquals(rows.collectAsList(), read.collectAsList());
    }

    /** Concatenates the messages of the whole cause chain, since Spark may wrap data source exceptions. */
    private static String messagesOf(Throwable thrown) {
        StringBuilder sb = new StringBuilder();
        for (Throwable t = thrown; t != null; t = t.getCause()) {
            sb.append(t.getMessage()).append('\n');
        }
        return sb.toString();
    }
}
