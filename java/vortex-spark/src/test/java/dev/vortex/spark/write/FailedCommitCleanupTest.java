// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.api.VortexWriter;
import dev.vortex.io.NativeWritable;
import dev.vortex.spark.write.PartitionedVortexDataWriter.ResolvedTransform;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A task whose {@code commit()} fails is failed by Spark, which calls {@code abort()} and {@code close()} on its writer
 * and retries the task. No file of the failed attempt may survive that, or the retry's rows are duplicated and a
 * footer-less file breaks every later read.
 */
final class FailedCommitCleanupTest {

    private static final StructType SCHEMA = new StructType(new StructField[] {
        new StructField("p", DataTypes.IntegerType, false, Metadata.empty()),
        new StructField("v", DataTypes.IntegerType, false, Metadata.empty())
    });
    private static final CaseInsensitiveStringMap OPTIONS = new CaseInsensitiveStringMap(Map.of());

    @TempDir
    Path tempDir;

    /**
     * Opens native writers that stream into the file at their {@code file://} URL, like the production writer, except
     * that the sink of the file whose path contains {@code failingPathPart} fails when the writer flushes it at finish.
     */
    private static VortexDataWriter.WriterOpener failingAtFinish(String failingPathPart) {
        return (session, filePath, arrowSchema, allocator, options) -> {
            Path file = Path.of(URI.create(filePath));
            Files.createDirectories(file.getParent());
            Files.createFile(file);
            boolean fails = filePath.contains(failingPathPart);
            NativeWritable sink = new NativeWritable() {
                @Override
                public void write(byte[] buffer, int offset, int length) throws IOException {
                    try (OutputStream out = Files.newOutputStream(file, StandardOpenOption.APPEND)) {
                        out.write(buffer, offset, length);
                    }
                }

                @Override
                public void flush() throws IOException {
                    if (fails) {
                        throw new IOException("injected failure finishing " + filePath);
                    }
                }

                @Override
                public void close() {}
            };
            return VortexWriter.builder(session, sink, arrowSchema, allocator).build();
        };
    }

    private static InternalRow row(int p, int v) {
        return new GenericInternalRow(new Object[] {p, v});
    }

    private List<Path> vortexFiles() throws IOException {
        try (Stream<Path> files = Files.walk(tempDir)) {
            return files.filter(f -> f.toString().endsWith(".vortex")).collect(Collectors.toList());
        }
    }

    @Test
    void abortAfterFailedCommitDeletesTheFile() throws IOException {
        Path file = tempDir.resolve("part-00000-0.vortex");
        VortexDataWriter writer = new VortexDataWriter(
                file.toUri().toString(),
                SCHEMA,
                OPTIONS,
                failingAtFinish(file.getFileName().toString()));
        writer.write(row(1, 1));

        assertThrows(IOException.class, writer::commit);
        writer.abort();
        writer.close();

        assertFalse(Files.exists(file), "the failed attempt's file survived");
    }

    @Test
    void failedCommitLeavesNoFileOfAnyPartition() throws IOException {
        ResolvedTransform[] transforms =
                PartitionedVortexDataWriter.resolveTransforms(new Transform[] {Expressions.identity("p")}, SCHEMA);
        PartitionedVortexDataWriter writer = new PartitionedVortexDataWriter(
                tempDir.toUri().toString(), SCHEMA, OPTIONS, transforms, 0, 0, failingAtFinish("/p=2/"));
        writer.write(row(1, 1));
        writer.write(row(2, 2));

        assertThrows(IOException.class, writer::commit);
        writer.abort();
        writer.close();

        assertEquals(List.of(), vortexFiles(), "files of the failed attempt survived");
    }
}
