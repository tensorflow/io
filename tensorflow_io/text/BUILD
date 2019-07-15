licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_io:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "text_ops",
    srcs = [
        "kernels/csv_output.cc",
        "kernels/text_input.cc",
        "kernels/text_output.cc",
        "kernels/text_sequence.cc",
        "ops/text_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core:output_ops",
        "//tensorflow_io/core:sequence_ops",
    ],
)
