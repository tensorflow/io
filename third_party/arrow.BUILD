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
        "format/Tensor.fbs",
    ],
    flatc_args = [
        "--no-union-value-namespacing",
        "--gen-object-api",
    ],
    out_prefix = "cpp/src/arrow/ipc/",
)

cc_library(
    name = "arrow",
    srcs = glob(
        [
            "cpp/src/arrow/*.cc",
            "cpp/src/arrow/*.h",
            "cpp/src/arrow/adapters/tensorflow/convert.h",
            "cpp/src/arrow/io/*.cc",
            "cpp/src/arrow/io/*.h",
            "cpp/src/arrow/ipc/*.cc",
            "cpp/src/arrow/ipc/*.h",
            "cpp/src/arrow/util/*.cc",
            "cpp/src/arrow/util/*.h",
        ],
        exclude = [
            "cpp/src/arrow/**/*-test.cc",
            "cpp/src/arrow/**/*benchmark*.cc",
            "cpp/src/arrow/**/*hdfs*.cc",
            "cpp/src/arrow/util/compression_zstd.*",
            "cpp/src/arrow/util/compression_lz4.*",
            "cpp/src/arrow/util/compression_brotli.*",
            "cpp/src/arrow/ipc/json*.cc",
            "cpp/src/arrow/ipc/stream-to-file.cc",
            "cpp/src/arrow/ipc/file-to-stream.cc",
        ],
    ) + [
        "cpp/src/parquet/api/io.h",
        "cpp/src/parquet/api/reader.h",
        "cpp/src/parquet/api/schema.h",
        "cpp/src/parquet/encoding.h",
        "cpp/src/parquet/encoding-internal.h",
        "cpp/src/parquet/exception.h",
        "cpp/src/parquet/schema-internal.h",
        "cpp/src/parquet/thrift.h",
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
        "cpp/src/parquet/util/comparison.cc",
        "cpp/src/parquet/util/comparison.h",
        "cpp/src/parquet/util/memory.cc",
        "cpp/src/parquet/util/memory.h",
        "cpp/src/parquet/util/visibility.h",
        "cpp/src/parquet/util/macros.h",
        "cpp/src/parquet/util/windows_compatibility.h",
        "cpp/src/parquet/parquet_version.h",
        "cpp/src/parquet/parquet_types.cpp",
        "cpp/src/parquet/parquet_types.h",
    ],
    hdrs = [
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    defines = [
        "ARROW_WITH_SNAPPY",
    ],
    includes = [
        "cpp/src",
    ],
    deps = [
        ":arrow_format",
        "@boost",
        "@snappy",
        "@thrift",
        "@zlib",
    ],
)
