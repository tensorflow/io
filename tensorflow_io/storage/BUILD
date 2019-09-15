licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "storage_ops",
    srcs = [
        "kernels/storage_kernels.cc",
        "kernels/storage_s3_kernels.cc",
        "ops/storage_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/kinesis:kinesis_ops",
    ],
)
