# Cloud BigQuery client for TensorFlow
licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

py_library(
    name = "bigquery",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
        "python/ops/bigquery_api.py",
    ],
    data = [
        "//tensorflow_io/core:python/ops/libtensorflow_io.so",
    ],
    srcs_version = "PY2AND3",
)

# Have to put test op into prod binary because of https://github.com/grpc/grpc/issues/20034
# Otherwise whenever 2 dynamically built libraries that are using grpc loaded
# using _load_library, only first one will work.
cc_library(
    name = "bigquery_ops",
    srcs = [
        "kernels/bigquery_dataset_op.cc",
        "kernels/bigquery_kernels.cc",
        "kernels/test_kernels/bigquery_test_client_op.cc",
        "ops/bigquery_ops.cc",
        "ops/bigquery_test_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        ":bigquery_lib_cc",
        "@avro",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googleapis//google/cloud/bigquery/storage/v1beta1:bigquerystorage_cc_grpc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

# A library for use in the bigquery kernels.
cc_library(
    name = "bigquery_lib_cc",
    srcs = ["kernels/bigquery_lib.cc"],
    hdrs = ["kernels/bigquery_lib.h"],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@avro",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googleapis//google/cloud/bigquery/storage/v1beta1:bigquerystorage_cc_grpc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
