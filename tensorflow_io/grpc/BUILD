licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//tensorflow_io:tensorflow_io.bzl",
    "tf_io_copts",
)

genrule(
    name = "endpoint_py",
    srcs = [
        "endpoint.proto",
    ],
    outs = [
        "python/ops/__init__.py",
        "python/ops/endpoint_pb2.py",
        "python/ops/endpoint_pb2_grpc.py",
    ],
    cmd = "python -m grpc_tools.protoc -Itensorflow_io/grpc --python_out=$(BINDIR)/tensorflow_io/grpc/python/ops/ --grpc_python_out=$(BINDIR)/tensorflow_io/grpc/python/ops/ $< ; touch $(BINDIR)/tensorflow_io/grpc/python/ops/__init__.py",
    output_to_bindir = True,
)

proto_library(
    name = "_any_proto_only",
    deps = ["@com_google_protobuf//:any_proto"],
)

cc_proto_library(
    name = "any_proto",
    deps = ["@com_google_protobuf//:any_proto"],
)

cc_grpc_library(
    name = "endpoint_cc",
    srcs = [
        "endpoint.proto",
    ],
    proto_only = False,
    well_known_protos = True,
    deps = [":any_proto"],
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
        ":endpoint_cc",
        "//tensorflow_io/core:dataset_ops",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_protobuf//:protobuf",
        "@libarchive",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
