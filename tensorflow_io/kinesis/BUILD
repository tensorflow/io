licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "kinesis_ops",
    srcs = [
        "kernels/kinesis_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@aws",
        "@curl",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
