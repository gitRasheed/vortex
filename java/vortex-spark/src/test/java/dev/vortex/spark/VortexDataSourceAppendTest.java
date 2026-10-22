// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Appends from separate Spark applications, each with its own SparkContext. */
public final class VortexDataSourceAppendTest {

    @TempDir
    Path tempDir;

    /**
     * Two applications appending to the same directory must keep each other's files. Task attempt ids restart at zero
     * for every SparkContext, so a file name derived from the partition and task ids alone is reused by the second
     * application and silently replaces the first one's data.
     */
    @Test
    public void appendsFromTwoSparkContextsKeepBothFiles() throws IOException {
        String flat = tempDir.resolve("flat").toUri().toString();
        String partitioned = tempDir.resolve("partitioned").toUri().toString();

        appendRange(flat, partitioned, 0).stop();
        SparkSession spark = appendRange(flat, partitioned, 100);
        try {
            assertEquals(2, countVortexFiles(tempDir.resolve("flat")));
            assertEquals(20, spark.read().format("vortex").load(flat).count());
            assertEquals(4, countVortexFiles(tempDir.resolve("partitioned")));
            assertEquals(20, spark.read().format("vortex").load(partitioned).count());
        } finally {
            spark.stop();
        }
    }

    private static SparkSession appendRange(String flat, String partitioned, int start) {
        SparkSession spark = SparkSession.builder()
                .appName("VortexDataSourceAppendTest-" + start)
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        var df = spark.range(start, start + 10)
                .selectExpr("id", "cast(id % 2 as int) as p")
                .coalesce(1);
        df.write().format("vortex").mode(SaveMode.Append).save(flat);
        df.write().format("vortex").partitionBy("p").mode(SaveMode.Append).save(partitioned);
        return spark;
    }

    private static long countVortexFiles(Path dir) throws IOException {
        try (Stream<Path> files = Files.walk(dir)) {
            return files.filter(f -> f.toString().endsWith(".vortex")).count();
        }
    }
}
