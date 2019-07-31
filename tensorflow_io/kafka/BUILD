licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_binary(
    name = "python/ops/_kafka_ops.so",
    srcs = [
        "kernels/kafka_dataset_ops.cc",
        "kernels/kafka_sequence.cc",
        "ops/dataset_ops.cc",
        "ops/kafka_ops.cc",
    ],
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:sequence_ops",
        "@kafka",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
