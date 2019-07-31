licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_binary(
    name = "python/ops/_libsvm_ops.so",
    srcs = [
        "kernels/decode_libsvm_op.cc",
        "ops/libsvm_ops.cc",
    ],
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
