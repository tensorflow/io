licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_io:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "ffmpeg_3.4_ops",
    srcs = [
        "kernels/audio_ffmpeg_input.cc",
        "kernels/audio_ffmpeg_reader.cc",
        "kernels/audio_ffmpeg_reader.h",
        "kernels/video_ffmpeg_input.cc",
        "kernels/video_ffmpeg_reader.cc",
        "kernels/video_ffmpeg_reader.h",
        "ops/ffmpeg_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = ["."],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:ffmpeg_3.4",
    ],
)

cc_library(
    name = "ffmpeg_2.8_ops",
    srcs = [
        "kernels/audio_ffmpeg_input.cc",
        "kernels/audio_ffmpeg_reader.cc",
        "kernels/audio_ffmpeg_reader.h",
        "kernels/video_ffmpeg_input.cc",
        "kernels/video_ffmpeg_reader.cc",
        "kernels/video_ffmpeg_reader.h",
        "ops/ffmpeg_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = ["."],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:ffmpeg_2.8",
    ],
)

cc_library(
    name = "libav_9.20_ops",
    srcs = [
        "kernels/audio_ffmpeg_input.cc",
        "kernels/audio_ffmpeg_reader.cc",
        "kernels/audio_ffmpeg_reader.h",
        "kernels/video_ffmpeg_input.cc",
        "kernels/video_ffmpeg_reader.cc",
        "kernels/video_ffmpeg_reader.h",
        "ops/ffmpeg_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = ["."],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:libav_9.20",
    ],
)
