licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_video_ops_ffmpeg_3.4.so',
    srcs = [
        "kernels/ffmpeg_3_4.cc",
        "kernels/video_reader.h",
        "kernels/video_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    includes = ["."],
    linkshared = 1,
    linkopts = [],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
	"@ffmpeg_3_4//:ffmpeg",
    ],
    data = [],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

cc_binary(
    name = 'python/ops/_video_ops_ffmpeg_2.8.so',
    srcs = [
        "kernels/ffmpeg_2_8.cc",
        "kernels/video_reader.h",
        "kernels/video_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    includes = ["."],
    linkshared = 1,
    linkopts = [],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
	"@ffmpeg_2_8//:ffmpeg",
    ],
    data = [],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

cc_binary(
    name = 'python/ops/_video_ops_libav_9.20.so',
    srcs = [
        "kernels/libav_9_20.cc",
        "kernels/video_reader.h",
        "kernels/video_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    includes = ["."],
    linkshared = 1,
    linkopts = [],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
	"@libav_9_20//:libav",
    ],
    data = [],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)
