licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "json_ops",
    srcs = [
        "kernels/json_kernels.cc",
        "ops/json_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/arrow:arrow_ops",
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = 1,
)
