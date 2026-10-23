// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.base.Splitter;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.apache.spark.unsafe.types.UTF8String;

/** Utilities for discovering and materializing Hive-style partition columns from file paths. */
public final class PartitionPathUtils {
    private static final String HIVE_DEFAULT_PARTITION = "__HIVE_DEFAULT_PARTITION__";
    /** Spark's file index option naming the directory partition discovery starts at for every path. */
    private static final String BASE_PATH_OPTION = "basePath";

    private static final Pattern URI_SCHEME = Pattern.compile("^[A-Za-z][A-Za-z0-9+.-]+:");
    private static final Splitter PATH_SPLITTER = Splitter.on('/').omitEmptyStrings();

    private PartitionPathUtils() {}

    /**
     * Parses the Hive-style {@code key=value} directories of a file path that lie below its table root (see
     * {@link #tableRoot}). As in Spark's own partition discovery, directories at or above the root and the file name
     * itself are not partition levels, so a table may live under a {@code key=value} directory without gaining a
     * column.
     *
     * @return an ordered map of partition column names to their string values
     */
    public static Map<String, String> parsePartitionValues(String tableRoot, String filePath) {
        List<String> segments = pathSegments(filePath);
        int from = Math.min(pathSegments(tableRoot).size(), segments.size());
        int to = Math.max(from, segments.size() - 1);
        return parsePartitionDirectories(segments.subList(from, to));
    }

    private static Map<String, String> parsePartitionDirectories(List<String> directories) {
        LinkedHashMap<String, String> values = new LinkedHashMap<>();
        for (String segment : directories) {
            int eqIdx = segment.indexOf('=');
            if (eqIdx > 0 && eqIdx < segment.length() - 1) {
                String key = URLDecoder.decode(segment.substring(0, eqIdx), StandardCharsets.UTF_8);
                String val = URLDecoder.decode(segment.substring(eqIdx + 1), StandardCharsets.UTF_8);
                values.put(key, val);
            }
        }
        return values;
    }

    /**
     * Returns the directory that partition discovery of {@code path} starts at, chosen as Spark's file index chooses
     * it: the {@code basePath} option when given, which must be an ancestor of every path read, otherwise the path
     * itself, or its parent directory when the path names a file.
     *
     * @throws IllegalArgumentException if the {@code basePath} option is not an ancestor of {@code path}
     */
    public static String tableRoot(String path, Map<String, String> options) {
        String basePath = new CaseInsensitiveStringMap(options).get(BASE_PATH_OPTION);
        if (basePath == null) {
            return path.endsWith(".vortex") ? parentDirectory(path) : path;
        }
        List<String> base = pathSegments(basePath);
        List<String> segments = pathSegments(path);
        if (segments.size() < base.size() || !segments.subList(0, base.size()).equals(base)) {
            throw new IllegalArgumentException(
                    String.format("Wrong %s %s for the root path: %s", BASE_PATH_OPTION, basePath, path));
        }
        return basePath;
    }

    private static String parentDirectory(String path) {
        int slash = path.lastIndexOf('/');
        return slash < 0 ? "" : path.substring(0, Math.max(slash, 1));
    }

    /**
     * Splits the path of a URI, or of a bare filesystem path, into its components. A bare path is made absolute and
     * {@code .}/{@code ..} are resolved, matching how the native listing normalizes its root, so that the depth of a
     * root and of the URIs listed under it line up.
     */
    static List<String> pathSegments(String uriOrPath) {
        String path;
        Matcher scheme = URI_SCHEME.matcher(uriOrPath);
        if (scheme.find()) {
            path = uriOrPath.substring(scheme.end());
            if (path.startsWith("//")) {
                int pathStart = path.indexOf('/', 2);
                path = pathStart < 0 ? "" : path.substring(pathStart);
            }
        } else {
            path = Paths.get(uriOrPath).toAbsolutePath().toString();
        }
        List<String> segments = new ArrayList<>();
        for (String segment : PATH_SPLITTER.split(path)) {
            if (segment.equals("..")) {
                if (!segments.isEmpty()) {
                    segments.remove(segments.size() - 1);
                }
            } else if (!segment.equals(".")) {
                segments.add(segment);
            }
        }
        return segments;
    }

    /**
     * Returns the partition column names shared by every file, in path order.
     *
     * <p>Like Spark's own partition discovery (though names are compared case-sensitively here), every file must sit
     * under the same {@code key=value} directory levels. A file outside the partition directories, or one missing a
     * level, would otherwise be read with a different column layout than the table schema describes.
     *
     * @param partitionValuesPerFile each file's partition values (see {@link #parsePartitionValues}) keyed by its path
     * @throws IllegalArgumentException if the files disagree on their partition columns
     */
    public static List<String> partitionColumnNames(Map<String, Map<String, String>> partitionValuesPerFile) {
        List<String> expected = null;
        String expectedPath = null;
        for (Map.Entry<String, Map<String, String>> file : partitionValuesPerFile.entrySet()) {
            String path = file.getKey();
            List<String> names = List.copyOf(file.getValue().keySet());
            if (expected == null) {
                expected = names;
                expectedPath = path;
            } else if (!expected.equals(names)) {
                throw new IllegalArgumentException(String.format(
                        "Conflicting partition column names detected: %s has partition columns %s but %s has %s. "
                                + "Every file of a partitioned table must sit under the same key=value directory "
                                + "levels.",
                        expectedPath, expected, path, names));
            }
        }
        return expected == null ? List.of() : expected;
    }

    /**
     * Discovers the partition columns of a set of files and infers each column's type from the values of every file.
     *
     * @param partitionValuesPerFile each file's partition values (see {@link #parsePartitionValues}) keyed by its path
     * @return the partition column names in path order, mapped to their inferred types
     * @throws IllegalArgumentException if the files disagree on their partition columns
     */
    public static Map<String, DataType> inferPartitionColumnTypes(
            Map<String, Map<String, String>> partitionValuesPerFile) {
        Collection<Map<String, String>> valuesPerFile = partitionValuesPerFile.values();
        Map<String, DataType> types = new LinkedHashMap<>();
        for (String name : partitionColumnNames(partitionValuesPerFile)) {
            List<String> values = valuesPerFile.stream()
                    .map(fileValues -> fileValues.get(name))
                    .collect(Collectors.toList());
            types.put(name, inferCommonPartitionColumnType(values));
        }
        return types;
    }

    /**
     * Single-value form of {@link #inferCommonPartitionColumnType(Collection)}: infers a Spark {@link DataType} from
     * one partition value string.
     */
    public static DataType inferPartitionColumnType(String value) {
        return inferCommonPartitionColumnType(Collections.singletonList(value));
    }

    /**
     * Infers the Spark {@link DataType} of a partition column from its values across every file. Each value is
     * classified on its own as integer, long, double, boolean, string, or null ({@code __HIVE_DEFAULT_PARTITION__}),
     * and the classes are widened pairwise like Spark's {@code PartitioningUtils.findWiderTypeForPartitionColumn}: null
     * widens to any other type, integer widens to long or double, and every other mix falls back to string, in
     * particular long with double, whose values a double cannot hold exactly. A column of only null values is a string.
     */
    public static DataType inferCommonPartitionColumnType(Collection<String> values) {
        DataType common = DataTypes.NullType;
        for (String value : values) {
            common = widenPartitionColumnType(common, classifyPartitionValue(value));
        }
        return DataTypes.NullType.equals(common) ? DataTypes.StringType : common;
    }

    private static DataType classifyPartitionValue(String value) {
        if (value == null || HIVE_DEFAULT_PARTITION.equals(value)) {
            return DataTypes.NullType;
        }
        if (Ints.tryParse(value) != null) {
            return DataTypes.IntegerType;
        }
        if (Longs.tryParse(value) != null) {
            return DataTypes.LongType;
        }
        if (Doubles.tryParse(value) != null) {
            return DataTypes.DoubleType;
        }
        if ("true".equalsIgnoreCase(value) || "false".equalsIgnoreCase(value)) {
            return DataTypes.BooleanType;
        }
        return DataTypes.StringType;
    }

    /** Spark's partition column widening rule, restricted to the types {@link #classifyPartitionValue} produces. */
    private static DataType widenPartitionColumnType(DataType left, DataType right) {
        if (left.equals(right) || DataTypes.NullType.equals(right)) {
            return left;
        }
        if (DataTypes.NullType.equals(left)) {
            return right;
        }
        if (isPair(left, right, DataTypes.IntegerType, DataTypes.LongType)) {
            return DataTypes.LongType;
        }
        if (isPair(left, right, DataTypes.IntegerType, DataTypes.DoubleType)) {
            return DataTypes.DoubleType;
        }
        return DataTypes.StringType;
    }

    private static boolean isPair(DataType left, DataType right, DataType first, DataType second) {
        return (left.equals(first) && right.equals(second)) || (left.equals(second) && right.equals(first));
    }

    /**
     * Creates a Spark {@link ConstantColumnVector} populated with the given partition value, parsed according to the
     * target {@link DataType}.
     */
    public static ConstantColumnVector createConstantVector(int numRows, DataType type, String value) {
        ConstantColumnVector vec = new ConstantColumnVector(numRows, type);
        if (value == null || HIVE_DEFAULT_PARTITION.equals(value)) {
            vec.setNull();
            return vec;
        }
        vec.setNotNull();
        if (type instanceof StringType) {
            vec.setUtf8String(UTF8String.fromString(value));
        } else if (type instanceof IntegerType || type instanceof DateType) {
            vec.setInt(Integer.parseInt(value));
        } else if (type instanceof LongType || type instanceof TimestampType || type instanceof TimestampNTZType) {
            vec.setLong(Long.parseLong(value));
        } else if (type instanceof ShortType) {
            vec.setShort(Short.parseShort(value));
        } else if (type instanceof ByteType) {
            vec.setByte(Byte.parseByte(value));
        } else if (type instanceof BooleanType) {
            vec.setBoolean(Boolean.parseBoolean(value));
        } else if (type instanceof FloatType) {
            vec.setFloat(Float.parseFloat(value));
        } else if (type instanceof DoubleType) {
            vec.setDouble(Double.parseDouble(value));
        } else {
            vec.setUtf8String(UTF8String.fromString(value));
        }
        return vec;
    }
}
