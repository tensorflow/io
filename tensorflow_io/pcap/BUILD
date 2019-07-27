licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_io:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_binary(
    name = "python/ops/_pcap_ops.so",
    srcs = [
        "kernels/pcap_input.cc",
        "ops/pcap_ops.cc",
    ],
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
