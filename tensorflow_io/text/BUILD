licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_text_ops.so",
    srcs = [
        "kernels/text_input.cc",
        "kernels/text_sequence.cc",
        "ops/text_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = [
        ".",
    ],
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core:sequence_ops",
        "@libarchive",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
