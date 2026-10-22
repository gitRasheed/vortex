// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.base.Splitter;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;
import org.apache.spark.sql.catalyst.catalog.ExternalCatalogUtils;
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
import org.apache.spark.unsafe.types.UTF8String;

/** Utilities for discovering and materializing Hive-style partition columns from file paths. */
public final class PartitionPathUtils {
    private static final String HIVE_DEFAULT_PARTITION = "__HIVE_DEFAULT_PARTITION__";
    private static final Splitter PATH_SPLITTER = Splitter.on('/');
    // A scheme of at least two characters, as the native layer requires: `C:\dir` is a path.
    private static final Pattern URL_SCHEME = Pattern.compile("^[A-Za-z][A-Za-z0-9+.-]+:");

    private PartitionPathUtils() {}

    /**
     * Parses Hive-style {@code key=value} segments from a file path.
     *
     * <p>The segments of a URL, such as the native listing returns, are percent-encoded and are decoded first; a bare
     * path is literal. Keys and values are then unescaped the way Spark's own sources escape them on write
     * ({@link ExternalCatalogUtils#unescapePathName}), so the two layouts are interchangeable.
     *
     * @return an ordered map of partition column names to their string values
     */
    public static Map<String, String> parsePartitionValues(String filePath) {
        boolean url = isUrl(filePath);
        LinkedHashMap<String, String> values = new LinkedHashMap<>();
        for (String segment : PATH_SPLITTER.split(filePath)) {
            String directory = url ? percentDecode(segment) : segment;
            int eqIdx = directory.indexOf('=');
            if (eqIdx > 0 && eqIdx < directory.length() - 1) {
                String key = ExternalCatalogUtils.unescapePathName(directory.substring(0, eqIdx));
                String val = ExternalCatalogUtils.unescapePathName(directory.substring(eqIdx + 1));
                values.put(key, val);
            }
        }
        return values;
    }

    /**
     * Whether the native layer reads {@code path} as a URL, whose path is percent-encoded, rather than as a literal
     * file path.
     */
    public static boolean isUrl(String path) {
        return URL_SCHEME.matcher(path).find();
    }

    /**
     * Decodes the {@code %XX} escapes of a URL path segment as UTF-8 and leaves every other character alone. Unlike
     * {@link java.net.URLDecoder}, which decodes HTML form data, a {@code +} stays a {@code +}.
     */
    private static String percentDecode(String segment) {
        if (segment.indexOf('%') < 0) {
            return segment;
        }
        byte[] encoded = segment.getBytes(StandardCharsets.UTF_8);
        ByteArrayOutputStream decoded = new ByteArrayOutputStream(encoded.length);
        for (int i = 0; i < encoded.length; i++) {
            if (encoded[i] == '%' && i + 2 < encoded.length) {
                int hi = Character.digit(encoded[i + 1] & 0xFF, 16);
                int lo = Character.digit(encoded[i + 2] & 0xFF, 16);
                if (hi >= 0 && lo >= 0) {
                    decoded.write((hi << 4) | lo);
                    i += 2;
                    continue;
                }
            }
            decoded.write(encoded[i]);
        }
        return decoded.toString(StandardCharsets.UTF_8);
    }

    /**
     * Infers a Spark {@link DataType} from a partition value string. Tries integer, long, double, boolean, and falls
     * back to string.
     */
    public static DataType inferPartitionColumnType(String value) {
        if (value == null || HIVE_DEFAULT_PARTITION.equals(value)) {
            return DataTypes.StringType;
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
