# Cloud Bigtable client for TensorFlow
licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "python/ops/_bigtable.so",
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
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        ":bigtable_lib_cc",
        "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "python/ops/_bigtable_test.so",
    srcs = [
        "kernels/test_kernels/bigtable_test_client_op.cc",
        "ops/bigtable_test_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        ":bigtable_lib_cc",
        ":bigtable_test_client",
        "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

# using cc_test in tensorflow_io currently results in segmentation fault. Using cc_binary is an interim solution.
cc_binary(
    name = "bigtable_test_client_test",
    srcs = [
        "kernels/test_kernels/bigtable_test_client_test.cc",
    ],
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
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkstatic = True,
    deps = [
        "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
        "@com_googlesource_code_re2//:re2",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

# A library for use in the bigtable kernels.
cc_library(
    name = "bigtable_lib_cc",
    srcs = ["kernels/bigtable_lib.cc"],
    hdrs = ["kernels/bigtable_lib.h"],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkstatic = True,
    deps = [
        "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
