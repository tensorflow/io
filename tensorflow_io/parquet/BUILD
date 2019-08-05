licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "parquet_ops",
    srcs = [
        "kernels/parquet_kernels.cc",
        "ops/parquet_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/arrow:arrow_ops",
        "//tensorflow_io/core:dataset_ops",
    ],
)
