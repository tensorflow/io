licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_parquet_ops.so",
    srcs = [
        "kernels/parquet_input.cc",
        "ops/parquet_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
