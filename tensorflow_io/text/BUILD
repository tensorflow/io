licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_text_ops.so",
    srcs = [
        "kernels/sequence_ops.h",
        "kernels/text_sequence.cc",
        "ops/text_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-DNDEBUG",
    ],
    includes = [
        ".",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
