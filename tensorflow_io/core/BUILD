licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//:tools/build/tensorflow_io.bzl", "tf_io_copts", "tf_io_features")

cc_library(
    name = "text_ops",
    srcs = [
        "kernels/text_re2.cc",
        "ops/text_ops.cc",
    ],
    copts = tf_io_copts(),
    features = tf_io_features(),
    linkstatic = True,
    deps = [
        "@com_googlesource_code_re2//:re2",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "python/ops/libtensorflow_io.so",
    copts = tf_io_copts(),
    features = tf_io_features(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:text_ops",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
