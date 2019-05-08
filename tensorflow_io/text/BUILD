licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "text_ops",
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
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core:sequence_ops",
    ],
)
