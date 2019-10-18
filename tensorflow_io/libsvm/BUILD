licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "libsvm_ops",
    srcs = [
        "kernels/decode_libsvm_op.cc",
        "ops/libsvm_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
