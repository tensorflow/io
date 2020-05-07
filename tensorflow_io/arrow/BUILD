licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "arrow_ops",
    srcs = [
        "kernels/arrow_dataset_ops.cc",
        "kernels/arrow_kernels.cc",
        "kernels/arrow_kernels.h",
        "kernels/arrow_stream_client.h",
        "ops/dataset_ops.cc",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "kernels/arrow_stream_client_windows.cc",
        ],
        "//conditions:default": [
            "kernels/arrow_stream_client_unix.cc",
        ],
    }),
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        ":arrow_util",
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
    ],
    alwayslink = 1,
)

cc_library(
    name = "arrow_util",
    srcs = ["kernels/arrow_util.cc"],
    hdrs = ["kernels/arrow_util.h"],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
    ],
    alwayslink = 1,
)
