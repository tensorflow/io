licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "audio_ops",
    srcs = [
        "kernels/audio_input.cc",
        "ops/audio_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
    ],
)
