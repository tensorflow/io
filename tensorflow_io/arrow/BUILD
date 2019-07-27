licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_io:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_binary(
    name = "python/ops/_arrow_ops.so",
    srcs = [
        "kernels/arrow_dataset_ops.cc",
        "kernels/arrow_stream_client.h",
        "kernels/arrow_stream_client_unix.cc",
        "kernels/arrow_util.cc",
        "kernels/arrow_util.h",
        "ops/dataset_ops.cc",
    ],
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "@arrow",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
