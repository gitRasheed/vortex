// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
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

    /** Concatenates the messages of the whole cause chain, since Spark may wrap data source exceptions. */
    private static String messagesOf(Throwable thrown) {
        StringBuilder sb = new StringBuilder();
        for (Throwable t = thrown; t != null; t = t.getCause()) {
            sb.append(t.getMessage()).append('\n');
        }
        return sb.toString();
    }
}
