licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_binary(
    name = "python/ops/_parquet_ops.so",
    srcs = [
        "kernels/parquet_input.cc",
        "ops/parquet_ops.cc",
    ],
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
