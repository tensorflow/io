licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_video_ops_ffmpeg_3.4.so",
    srcs = [
        "kernels/ffmpeg.cc",
        "kernels/video_input.cc",
        "kernels/video_reader.h",
        "ops/video_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = ["."],
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_3_4//:ffmpeg",
    ],
)

cc_binary(
    name = "python/ops/_video_ops_ffmpeg_2.8.so",
    srcs = [
        "kernels/ffmpeg.cc",
        "kernels/video_input.cc",
        "kernels/video_reader.h",
        "ops/video_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = ["."],
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_2_8//:ffmpeg",
    ],
)

cc_binary(
    name = "python/ops/_video_ops_libav_9.20.so",
    srcs = [
        "kernels/ffmpeg.cc",
        "kernels/video_input.cc",
        "kernels/video_reader.h",
        "ops/video_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = ["."],
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@libav_9_20//:libav",
    ],
)
