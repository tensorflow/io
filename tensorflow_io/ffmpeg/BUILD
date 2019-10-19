licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "ffmpeg_3.4_ops",
    srcs = [
        "kernels/ffmpeg_kernels.cc",
        "ops/ffmpeg_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = ["."],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_3_4//:ffmpeg",
    ],
)

cc_library(
    name = "ffmpeg_2.8_ops",
    srcs = [
        "kernels/ffmpeg_kernels.cc",
        "ops/ffmpeg_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = ["."],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_2_8//:ffmpeg",
    ],
)
