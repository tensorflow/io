# Description:
#   Apache Arrow library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

flatbuffer_cc_library(
    name = "arrow_format",
    srcs = [
        "cpp/src/arrow/ipc/feather.fbs",
        "format/File.fbs",
        "format/Message.fbs",
        "format/Schema.fbs",
        "format/SparseTensor.fbs",
        "format/Tensor.fbs",
    ],
    flatc_args = [
        "--no-union-value-namespacing",
        "--gen-object-api",
    ],
    out_prefix = "cpp/src/arrow/ipc/",
)

genrule(
    name = "arrow_util_config",
    srcs = ["cpp/src/arrow/util/config.h.cmake"],
    outs = ["cpp/src/arrow/util/config.h"],
    cmd = "sed 's/@ARROW_VERSION_MAJOR@/0/g ; s/@ARROW_VERSION_MINOR@/14/g ; s/@ARROW_VERSION_PATCH@/1/g ; s/#cmakedefine/#undef/g' $< > $@",
)

cc_library(
    name = "arrow",
    srcs = glob(
        [
            "cpp/src/arrow/*.cc",
            "cpp/src/arrow/*.h",
            "cpp/src/arrow/adapters/tensorflow/convert.h",
            "cpp/src/arrow/array/*.cc",
            "cpp/src/arrow/array/*.h",
            "cpp/src/arrow/dataset/*.h",
            "cpp/src/arrow/io/*.cc",
            "cpp/src/arrow/io/*.h",
            "cpp/src/arrow/ipc/*.cc",
            "cpp/src/arrow/ipc/*.h",
            "cpp/src/arrow/util/*.cc",
            "cpp/src/arrow/util/*.h",
            "cpp/src/arrow/vendored/**/*.cpp",
            "cpp/src/arrow/vendored/**/*.c",
            "cpp/src/arrow/vendored/**/*.hpp",
            "cpp/src/arrow/vendored/**/*.h",
        ],
        exclude = [
            "cpp/src/arrow/**/*-test.cc",
            "cpp/src/arrow/**/*benchmark*.cc",
            "cpp/src/arrow/**/*hdfs*.cc",
            "cpp/src/arrow/util/compression_zstd.*",
            "cpp/src/arrow/util/compression_lz4.*",
            "cpp/src/arrow/util/compression_brotli.*",
            "cpp/src/arrow/util/compression_bz2.*",
            "cpp/src/arrow/util/uri.*",
            "cpp/src/arrow/util/ubsan.cc",
            "cpp/src/arrow/io/test-common.*",
            "cpp/src/arrow/ipc/json*.cc",
            "cpp/src/arrow/ipc/stream-to-file.cc",
            "cpp/src/arrow/ipc/file-to-stream.cc",
            "cpp/src/arrow/ipc/test-common.*",
        ],
    ) + [
        "cpp/src/parquet/api/io.h",
        "cpp/src/parquet/windows_compatibility.h",
        "cpp/src/parquet/api/reader.h",
        "cpp/src/parquet/api/schema.h",
        "cpp/src/parquet/deprecated_io.cc",
        "cpp/src/parquet/deprecated_io.h",
        "cpp/src/parquet/encoding.cc",
        "cpp/src/parquet/encoding.h",
        "cpp/src/parquet/exception.h",
        "cpp/src/parquet/schema-internal.h",
        "cpp/src/parquet/thrift.h",
        "cpp/src/parquet/platform.cc",
        "cpp/src/parquet/platform.h",
        "cpp/src/parquet/properties.cc",
        "cpp/src/parquet/properties.h",
        "cpp/src/parquet/column_page.h",
        "cpp/src/parquet/column_reader.cc",
        "cpp/src/parquet/column_reader.h",
        "cpp/src/parquet/column_scanner.cc",
        "cpp/src/parquet/column_scanner.h",
        "cpp/src/parquet/file_reader.cc",
        "cpp/src/parquet/file_reader.h",
        "cpp/src/parquet/metadata.cc",
        "cpp/src/parquet/metadata.h",
        "cpp/src/parquet/printer.cc",
        "cpp/src/parquet/printer.h",
        "cpp/src/parquet/schema.cc",
        "cpp/src/parquet/schema.h",
        "cpp/src/parquet/statistics.cc",
        "cpp/src/parquet/statistics.h",
        "cpp/src/parquet/types.cc",
        "cpp/src/parquet/types.h",
        "cpp/src/parquet/parquet_version.h",
        "cpp/src/parquet/parquet_types.cpp",
        "cpp/src/parquet/parquet_types.h",
    ],
    hdrs = [
        "cpp/src/arrow/util/config.h",  # declare header from above genrule
    ],
    defines = [
        "ARROW_WITH_SNAPPY",
    ],
    includes = [
        "cpp/src",
        "cpp/src/arrow/vendored/xxhash",
    ],
    deps = [
        ":arrow_format",
        "@boost",
        "@double_conversion//:double-conversion",
        "@snappy",
        "@thrift",
        "@zlib",
    ],
)
