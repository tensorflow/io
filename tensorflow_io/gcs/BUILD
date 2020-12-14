licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_binary(
    name = "_gcs_config_ops.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        ":gcs_config_ops",
    ],
)

cc_library(
    name = "gcs_config_ops",
    srcs = [
        "kernels/gcs_config_op_kernels.cc",
        "ops/gcs_config_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@curl",
        "@jsoncpp_git//:jsoncpp",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
