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

cc_library(
    name = "ffmpeg_3.4_ops",
    srcs = [
        "kernels/audio_ffmpeg_input.cc",
        "kernels/audio_ffmpeg_reader.cc",
        "kernels/audio_ffmpeg_reader.h",
        "ops/audio_ffmpeg_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
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
        "ops/audio_ffmpeg_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
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
        "ops/audio_ffmpeg_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = ["."],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:libav_9.20",
    ],
)
