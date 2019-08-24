licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "kafka_ops",
    srcs = [
        "kernels/kafka_dataset_ops.cc",
        "kernels/kafka_kernels.cc",
        "kernels/kafka_sequence.cc",
        "ops/dataset_ops.cc",
        "ops/kafka_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core:sequence_ops",
        "@kafka",
    ],
)
