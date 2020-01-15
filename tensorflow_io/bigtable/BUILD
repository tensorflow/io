# Cloud Bigtable client for TensorFlow
licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

py_library(
    name = "bigtable",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
        "python/ops/bigtable_api.py",
    ],
    data = [
        "//tensorflow_io/core:python/ops/libtensorflow_io.so",
    ],
    srcs_version = "PY2AND3",
)

cc_library(
    name = "bigtable_ops",
    srcs = [
        "kernels/bigtable_kernels.cc",
        "kernels/bigtable_lookup_dataset_op.cc",
        "kernels/bigtable_prefix_key_dataset_op.cc",
        "kernels/bigtable_range_helpers.cc",
        "kernels/bigtable_range_helpers.h",
        "kernels/bigtable_range_key_dataset_op.cc",
        "kernels/bigtable_sample_key_pairs_dataset_op.cc",
        "kernels/bigtable_sample_keys_dataset_op.cc",
        "kernels/bigtable_scan_dataset_op.cc",
        "ops/bigtable_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        ":bigtable_lib_cc",
        "@com_github_googleapis_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "bigtable_test",
    srcs = [
        "kernels/test_kernels/bigtable_test_client_op.cc",
        "ops/bigtable_test_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        ":bigtable_lib_cc",
        ":bigtable_test_client",
        "@com_github_googleapis_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

# using cc_test in tensorflow_io currently results in segmentation fault. Using cc_binary is an interim solution.
cc_binary(
    name = "bigtable_test_client_test",
    srcs = [
        "kernels/test_kernels/bigtable_test_client_test.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],
        "//conditions:default": ["-lrt"],
    }),
    deps = [
        ":bigtable_lib_cc",
        ":bigtable_test_client",
        "@com_google_googletest//:gtest_main",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "bigtable_test_client",
    srcs = [
        "kernels/test_kernels/bigtable_test_client.cc",
    ],
    hdrs = [
        "kernels/test_kernels/bigtable_test_client.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@com_github_googleapis_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
        "@com_googlesource_code_re2//:re2",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

# A library for use in the bigtable kernels.
cc_library(
    name = "bigtable_lib_cc",
    srcs = ["kernels/bigtable_lib.cc"],
    hdrs = ["kernels/bigtable_lib.h"],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@com_github_googleapis_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
