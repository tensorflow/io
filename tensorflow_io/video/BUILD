licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# Note: stub files are generated from Ubuntu 14.0 with
# `echo '' | g++ -shared -fPIC -x c++ - -o $@`
filegroup(
    name = "stub.so",
    srcs = glob(["kernel/stub/*"]),
)

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
    linkopts = [
        "-Ltensorflow_io/video/kernels/stub",
        "-l:libavformat.so.57",
        "-l:libavcodec.so.57",
        "-l:libavutil.so.55",
        "-l:libswscale.so.4",
    ],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
	"@ffmpeg_3_4//:ffmpeg",
    ],
    data = [
        ":stub.so",
    ],
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
    linkopts = [
        "-Ltensorflow_io/video/kernels/stub",
	"-l:libavformat-ffmpeg.so.56",
	"-l:libavcodec-ffmpeg.so.56",
	"-l:libavutil-ffmpeg.so.54",
	"-l:libswscale-ffmpeg.so.3",
    ],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
	"@ffmpeg_2_8//:ffmpeg",
    ],
    data = [
        ":stub.so",
    ],
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
    linkopts = [
        "-Ltensorflow_io/video/kernels/stub",
	"-l:libavformat.so.54",
	"-l:libavcodec.so.54",
	"-l:libavutil.so.52",
	"-l:libswscale.so.2",
    ],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
	"@libav_9_20//:libav",
    ],
    data = [
        ":stub.so",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)
