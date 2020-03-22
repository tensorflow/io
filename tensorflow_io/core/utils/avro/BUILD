licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)
load("@io_bazel_rules_go//go:def.bzl", "go_binary")


cc_library(
    name = "avro_utils_api",
    hdrs = ["prefix_tree.h",
            "value_buffer.h",
            "avro_parser.h",
            "avro_parser_tree.h",
            "parse_avro_attrs.h",
            "avro_file_stream_reader.h",
            "name_utils.h", # TODO(fraudies): delete when tensorflow/core/kernels/data/name_utils.h visible
            "avro_record_reader.h",
            ],
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "avro_utils",
    srcs = ["prefix_tree.cc",
            "value_buffer.cc",
            "avro_parser.cc",
            "avro_parser_tree.cc",
            "parse_avro_attrs.cc",
            "avro_file_stream_reader.cc",
            "name_utils.cc", # TODO(fraudies): delete when tensorflow/core/kernels/data/name_utils.h visible
            "avro_record_reader.cc",
            ],
    deps = [
        ":avro_utils_api",
        "@avro",
    ],
)
