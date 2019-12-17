# Description:
#   Nucleus library

licenses(["notice"])  # Apache 2.0 license

exports_files(["LICENSE"])

cc_library(
    name = "proto_ptr",
    hdrs = [
        "@nucleus//nucleus/util:proto_ptr.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "types",
    hdrs = ["@nucleus//nucleus/platform:types.h"],
)

cc_library(
    name = "statusor",
    hdrs = [
        "@nucleus//nucleus/vendor:statusor.h",
    ],
    deps = [
        ":types",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "hts_path",
    srcs = ["@nucleus//nucleus/io:hts_path.cc"],
    hdrs = ["@nucleus//nucleus/io:hts_path.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/strings",
        "@htslib",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "text_reader",
    srcs = ["@nucleus//nucleus/io:text_reader.cc"],
    hdrs = ["@nucleus//nucleus/io:text_reader.h"],
    deps = [
        ":hts_path",
        ":statusor",
        ":types",
        "@com_google_absl//absl/memory",
        "@htslib",
    ],
)

cc_library(
    name = "reader_base",
    srcs = ["@nucleus//nucleus/io:reader_base.cc"],
    hdrs = ["@nucleus//nucleus/io:reader_base.h"],
    deps = [
        ":proto_ptr",
        ":statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "cpp_utils",
    srcs = ["@nucleus//nucleus/util:utils.cc"],
    hdrs = ["@nucleus//nucleus/util:utils.h"],
    deps = [
        ":proto_ptr",
        ":types",
        "//nucleus/protos:reads_cc_pb2",
        "//nucleus/protos:variants_cc_pb2",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "fastq_reader",
    srcs = ["@nucleus//nucleus/io:fastq_reader.cc"],
    hdrs = ["@nucleus//nucleus/io:fastq_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cpp_utils",
        ":reader_base",
        ":text_reader",
        "@nucleus//nucleus/protos:fastq_cc_pb2",
    ],
)
