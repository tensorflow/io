licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

proto_library(
    name = "endpoint_proto",
    srcs = ["endpoint.proto"],
    deps = ["@com_google_protobuf//:any_proto"],
)

cc_proto_library(
    name = "endpoint_cc_proto",
    deps = [":endpoint_proto"],
)

cc_grpc_library(
    name = "endpoint_cc_grpc",
    srcs = [":endpoint_proto"],
    grpc_only = True,
    deps = [":endpoint_cc_proto"],
)

cc_binary(
    name = "python/ops/_grpc_ops.so",
    srcs = [
        "kernels/grpc_input.cc",
        "ops/grpc_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkshared = 1,
    deps = [
        ":endpoint_cc_grpc",
        "//tensorflow_io/core:dataset_ops",
        "@com_github_grpc_grpc//:grpc++",
        "@libarchive",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_proto_library(
    name = "endpoint_py_pb2",
    deps = [":endpoint_proto"],
)

py_grpc_library(
    name = "endpoint_py_pb2_grpc",
    srcs = [":endpoint_proto"],
    deps = [":endpoint_py_pb2"],
)
