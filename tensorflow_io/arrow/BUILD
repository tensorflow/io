licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "arrow_ops",
    srcs = [
        "kernels/arrow_dataset_ops.cc",
        "kernels/arrow_kernels.cc",
        "kernels/arrow_kernels.h",
        "kernels/arrow_stream_client.h",
        "kernels/arrow_stream_client_unix.cc",
        "kernels/arrow_util.cc",
        "kernels/arrow_util.h",
        "ops/dataset_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
    ],
)
