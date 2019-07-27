# Cloud BigQuery client for TensorFlow
licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//tensorflow_io:tensorflow_io.bzl",
    "tf_io_copts",
)

KERNEL_FILES = [
    "kernels/bigquery_kernels.cc",
    "kernels/bigquery_dataset_op.cc",
]

cc_binary(
    name = "python/ops/_bigquery.so",
    srcs = KERNEL_FILES + [
        "ops/bigquery_ops.cc",
    ],
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        ":bigquery_lib_cc",
        "@avro",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googleapis//google/cloud/bigquery/storage/v1beta1:bigquerystorage_cc_grpc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "python/kernel_tests/_bigquery_test.so",
    srcs = [
        "kernels/test_kernels/bigquery_test_client_op.cc",
        "ops/bigquery_test_ops.cc",
    ],
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        ":bigquery_fake",
        ":bigquery_lib_cc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googleapis//google/cloud/bigquery/storage/v1beta1:bigquerystorage_cc_grpc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "bigquery_fake_test",
    srcs = ["kernels/test_kernels/bigquery_fake_test.cc"],
    deps = [
        ":bigquery_fake",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googleapis//google/cloud/bigquery/storage/v1beta1:bigquerystorage_cc_grpc",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "bigquery_fake",
    srcs = ["kernels/test_kernels/fake_bigquery_storage_service.cc"],
    hdrs = ["kernels/test_kernels/fake_bigquery_storage_service.h"],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googleapis//google/cloud/bigquery/storage/v1beta1:bigquerystorage_cc_grpc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
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
)
