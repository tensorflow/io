licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "kinesis_ops",
    srcs = [
        "kernels/aws_kernels.cc",
        "kernels/kinesis_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    hdrs = [
        "kernels/aws_kernels.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@aws",
        "@curl",
    ],
)
