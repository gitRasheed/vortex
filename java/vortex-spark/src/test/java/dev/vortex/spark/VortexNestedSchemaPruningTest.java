// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import dev.vortex.spark.read.VortexScan;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that reading one field of a nested column returns that field, and that the struct fields Spark does not require
 * are pruned in the native scan.
 *
 * <p>With {@code spark.sql.optimizer.nestedSchemaPruning.enabled}, Spark asks the scan for only the nested fields a
 * query uses. Each row of the table holds {@code a = id + 100}, {@code b = id + 200} and {@code c = id + 300}, so a
 * field read from the wrong position shows up as the wrong hundred. Each query is checked against Parquet, and the
 * scan's read schema is checked to be the pruned one, so the results prove the pruned projection returns the right
 * fields rather than the whole struct.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexNestedSchemaPruningTest {

    private SparkSession spark;
    private String path;
    private String parquetPath;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexNestedSchemaPruningTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.optimizer.nestedSchemaPruning.enabled", "true")
                .config("spark.ui.enabled", "false")
                .getOrCreate();

        Column id = functions.col("id");
        Column abc = functions.struct(
                id.plus(100).alias("a"), id.plus(200).alias("b"), id.plus(300).alias("c"));
        Column xy = functions.struct(id.plus(10).alias("x"), id.plus(20).alias("y"));
        Dataset<Row> df = spark.range(5)
                .select(
                        id,
                        abc.alias("s"),
                        functions.array(abc).alias("arr"),
                        functions.map(functions.lit("k"), abc).alias("m"),
                        functions
                                .struct(functions.lit("x").alias("p"), abc.alias("inner"))
                                .alias("outer"),
                        functions
                                .when(id.mod(3).equalTo(0), functions.lit(null))
                                .otherwise(abc)
                                .alias("ns"),
                        functions
                                .when(id.mod(4).equalTo(0), functions.lit(null))
                                .otherwise(functions.struct(functions.lit("q").alias("q"), xy.alias("inner")))
                                .alias("nouter"),
                        functions
                                .struct(id.alias("n"), functions.array(abc, abc).alias("arr"))
                                .alias("sa"));

        path = tempDir.resolve("nested").toUri().toString();
        df.write().format("vortex").mode(SaveMode.Overwrite).save(path);
        parquetPath = tempDir.resolve("parquet").toUri().toString();
        df.write().format("parquet").mode(SaveMode.Overwrite).save(parquetPath);
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Selecting a struct field other than the first returns that field, without the other fields")
    public void structField() {
        assertSameAsParquet("struct<id:bigint,s:struct<c:bigint>>", "id", "s.c");
    }

    @Test
    @DisplayName("Selecting a field of the structs in an array returns that field, reading the array whole")
    public void arrayElementField() {
        assertSameAsParquet(
                "struct<id:bigint,arr:array<struct<a:bigint,b:bigint,c:bigint>>>", "id", "element_at(arr.c, 1)");
    }

    @Test
    @DisplayName("Selecting a field of the structs in a map returns that field, reading the map whole")
    public void mapValueField() {
        assertSameAsParquet("struct<id:bigint,m:map<string,struct<a:bigint,b:bigint,c:bigint>>>", "id", "m['k'].c");
    }

    @Test
    @DisplayName("Selecting a field of a struct inside a struct returns that field, pruned at both levels")
    public void structInStructField() {
        assertSameAsParquet("struct<id:bigint,outer:struct<inner:struct<c:bigint>>>", "id", "outer.inner.c");
    }

    @Test
    @DisplayName("A nullable struct keeps its null rows when pruned")
    public void nullableStruct() {
        assertSameAsParquet("struct<id:bigint,ns:struct<c:bigint>>", "id", "ns.c", "ns is null");
    }

    @Test
    @DisplayName("A nullable struct keeps its null rows when a struct inside it is pruned")
    public void nullableStructInStruct() {
        assertSameAsParquet(
                "struct<id:bigint,nouter:struct<inner:struct<x:bigint>>>", "id", "nouter.inner.x", "nouter is null");
    }

    @Test
    @DisplayName("An array inside a struct is read whole, without the struct's other fields")
    public void arrayInStruct() {
        assertSameAsParquet(
                "struct<id:bigint,sa:struct<arr:array<struct<a:bigint,b:bigint,c:bigint>>>>",
                "id",
                "element_at(sa.arr, 2).c");
    }

    @Test
    @DisplayName("A pushed filter on one field and a projection of another read only the projected field")
    public void filterOnAnotherField() {
        Dataset<Row> expected = load("parquet", null).filter("s.a > 102").selectExpr("id", "s.c");
        Dataset<Row> actual = load("vortex", null).filter("s.a > 102").selectExpr("id", "s.c");
        assertEquals(
                expected.orderBy("id").collectAsList(), actual.orderBy("id").collectAsList());
        assertEquals(
                "struct<id:bigint,s:struct<c:bigint>>", scanReadSchema(actual).simpleString());
    }

    @Test
    @DisplayName("A read schema that lists some struct fields, in its own order, reads each field by name")
    public void readSchemaBindsStructFieldsByName() {
        List<Row> rows = spark.read()
                .format("vortex")
                .schema("id bigint, s struct<c: bigint, a: bigint, z: bigint>, arr array<struct<c: bigint>>")
                .load(path)
                .orderBy("id")
                .selectExpr("s.c", "s.a", "s.z", "arr[0].c")
                .collectAsList();
        List<Row> expected = new ArrayList<>();
        for (long id = 0; id < 5; id++) {
            expected.add(RowFactory.create(300 + id, 100 + id, null, 300 + id));
        }
        assertEquals(expected, rows);
    }

    @Test
    @DisplayName("A read schema spelling a struct and its fields in another case is pruned by the file's names")
    public void readSchemaDifferingInCase() {
        String schema = "ID bigint, S struct<C: bigint, A: bigint>";
        assertSameAsParquetWithSchema(schema, "struct<ID:bigint,S:struct<C:bigint>>", "ID", "S.C");
        assertSameAsParquetWithSchema(schema, "struct<ID:bigint,S:struct<C:bigint,A:bigint>>", "ID", "S.A", "S.C");
    }

    @Test
    @DisplayName("A read schema with a struct field the file lacks reads that field as null")
    public void readSchemaFieldMissingFromFile() {
        String schema = "id bigint, s struct<c: bigint, z: bigint>";
        assertSameAsParquetWithSchema(schema, "struct<id:bigint,s:struct<z:bigint>>", "id", "s.z");
        assertSameAsParquetWithSchema(schema, "struct<id:bigint,s:struct<c:bigint,z:bigint>>", "id", "s.c", "s.z");
    }

    @Test
    @DisplayName("A collated string in the read schema reads a string field")
    public void readSchemaWithCollatedString() {
        assumeTrue(spark.version().startsWith("4."), "collations exist from Spark 4");
        List<Row> rows = spark.read()
                .format("vortex")
                .schema("id bigint, outer struct<p: string collate utf8_lcase>")
                .load(path)
                .selectExpr("outer.p")
                .collectAsList();
        assertEquals(
                List.of("x", "x", "x", "x", "x"),
                rows.stream().map(row -> row.getString(0)).toList());
    }

    private void assertSameAsParquet(String readSchema, String... exprs) {
        assertSameAsParquetWithSchema(null, readSchema, exprs);
    }

    private void assertSameAsParquetWithSchema(String schema, String readSchema, String... exprs) {
        Dataset<Row> expected = load("parquet", schema).selectExpr(exprs).orderBy("id");
        Dataset<Row> actual = load("vortex", schema).selectExpr(exprs).orderBy("id");
        assertEquals(expected.collectAsList(), actual.collectAsList());
        assertEquals(readSchema, scanReadSchema(actual).simpleString());
    }

    private Dataset<Row> load(String format, String schema) {
        DataFrameReader reader = spark.read().format(format);
        if (schema != null) {
            reader = reader.schema(schema);
        }
        return reader.load(format.equals("vortex") ? path : parquetPath);
    }

    private static StructType scanReadSchema(Dataset<Row> df) {
        BatchScanExec scan = (BatchScanExec) df.queryExecution()
                .executedPlan()
                .find(plan -> plan instanceof BatchScanExec)
                .get();
        return ((VortexScan) scan.scan()).readSchema();
    }
}
