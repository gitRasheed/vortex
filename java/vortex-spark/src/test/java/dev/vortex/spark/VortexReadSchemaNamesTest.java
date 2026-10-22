// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.List;
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
 * Tests that a read schema is matched against the files by name the way Spark's Parquet reader does: ignoring case
 * unless {@code spark.sql.caseSensitive} is set, and reading a column the file does not have as null, in projections
 * and in pushed-down filters alike.
 *
 * <p>The files hold {@code id} from 0 to 4 and {@code s} with {@code a = id + 100} and {@code c = id + 300}. A second
 * copy holds {@code id} partitioned by {@code p = id % 2}.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexReadSchemaNamesTest {

    private SparkSession spark;
    private String path;
    private String partitionedPath;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexReadSchemaNamesTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.ui.enabled", "false")
                .getOrCreate();

        path = tempDir.resolve("names").toUri().toString();
        spark.range(5)
                .select(
                        functions.col("id"),
                        functions
                                .struct(
                                        functions.col("id").plus(100).alias("a"),
                                        functions.col("id").plus(300).alias("c"))
                                .alias("s"))
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(path);

        partitionedPath = tempDir.resolve("partitioned").toUri().toString();
        spark.range(5)
                .withColumn("p", functions.col("id").mod(2))
                .write()
                .format("vortex")
                .partitionBy("p")
                .mode(SaveMode.Overwrite)
                .save(partitionedPath);
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Column and field names that differ from the file only in case resolve unless caseSensitive is set")
    public void namesDifferingInCase() {
        assertEquals(List.of(RowFactory.create(3L, 103L)), upperCaseQuery().collectAsList());

        // A new Dataset: the planned reader captures the setting.
        spark.conf().set("spark.sql.caseSensitive", "true");
        try {
            assertEquals(List.of(), upperCaseQuery().collectAsList(), "ID and S.C are not in the file, so are null");
        } finally {
            spark.conf().set("spark.sql.caseSensitive", "false");
        }
    }

    private Dataset<Row> upperCaseQuery() {
        return spark.read()
                .format("vortex")
                .schema("ID bigint, S struct<C: bigint, A: bigint>")
                .load(path)
                .where("S.C > 302 and ID < 4")
                .select("ID", "S.A");
    }

    @Test
    @DisplayName("A partition column declared in another case than its directories resolves like a data column")
    public void partitionColumnDifferingInCase() {
        Dataset<Row> table = spark.read()
                .format("vortex")
                .schema("ID bigint, P bigint")
                .load(partitionedPath)
                .where("P = 1");
        assertEquals(2, table.count());
        assertEquals(
                List.of(RowFactory.create(1L, 1L), RowFactory.create(3L, 1L)),
                table.select("ID", "P").orderBy("ID").collectAsList());
    }

    @Test
    @DisplayName("A column the file does not have reads as null")
    public void missingColumnReadsAsNull() {
        Dataset<Row> table = spark.read()
                .format("vortex")
                .schema("id bigint, zz bigint, s struct<a: bigint, zz: bigint>")
                .load(path);
        assertEquals(
                List.of(RowFactory.create(2L, null, null)),
                table.where("zz is null and s.zz is null and id = 2")
                        .select("id", "zz", "s.zz")
                        .collectAsList());
        assertEquals(0, table.where("zz > 1 or s.zz = 1").count());
    }
}
