licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)
load("@io_bazel_rules_go//go:def.bzl", "go_binary")

cc_library(
    name = "avro_utils_api",
    hdrs = [
        "avro_file_stream_reader.h",
        "avro_parser.h",
        "avro_parser_tree.h",
        "avro_record_reader.h",
        "name_utils.h",  # TODO(fraudies): delete when tensorflow/core/kernels/data/name_utils.h visible
        "parse_avro_attrs.h",
        "prefix_tree.h",
        "value_buffer.h",
    ],
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "avro_utils",
    srcs = [
        "avro_file_stream_reader.cc",
        "avro_parser.cc",
        "avro_parser_tree.cc",
        "avro_record_reader.cc",
        "name_utils.cc",  # TODO(fraudies): delete when tensorflow/core/kernels/data/name_utils.h visible
        "parse_avro_attrs.cc",
        "prefix_tree.cc",
        "value_buffer.cc",
    ],
    deps = [
        ":avro_utils_api",
        "@avro",
    ],
)