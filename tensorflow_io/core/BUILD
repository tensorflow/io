licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sequence_ops",
    srcs = [
        "kernels/sequence_ops.h",
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
        "@libarchive",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "dataset_ops",
    srcs = [
        "kernels/dataset_ops.h",
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
        "@libarchive",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "ffmpeg_3.4",
    srcs = [
        "kernels/ffmpeg_reader.cc",
        "kernels/ffmpeg_reader.h",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = ["."],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_3_4//:ffmpeg",
    ],
)

cc_library(
    name = "ffmpeg_2.8",
    srcs = [
        "kernels/ffmpeg_reader.cc",
        "kernels/ffmpeg_reader.h",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = ["."],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_2_8//:ffmpeg",
    ],
)

cc_library(
    name = "libav_9.20",
    srcs = [
        "kernels/ffmpeg_reader.cc",
        "kernels/ffmpeg_reader.h",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = ["."],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@libav_9_20//:libav",
    ],
)

cc_binary(
    name = "python/ops/libtensorflow_io.so",
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "//tensorflow_io/audio:audio_ops",
        "//tensorflow_io/avro:avro_ops",
        "//tensorflow_io/azure:azfs_ops",
        "//tensorflow_io/cifar:cifar_ops",
        "//tensorflow_io/hdf5:hdf5_ops",
        "//tensorflow_io/ignite:ignite_ops",
        "//tensorflow_io/lmdb:lmdb_ops",
        "//tensorflow_io/mnist:mnist_ops",
        "//tensorflow_io/prometheus:prometheus_ops",
        "//tensorflow_io/text:text_ops",
        "@libarchive",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "python/ops/libtensorflow_io_ffmpeg_3.4.so",
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "//tensorflow_io/ffmpeg:ffmpeg_3.4_ops",
    ],
)

cc_binary(
    name = "python/ops/libtensorflow_io_ffmpeg_2.8.so",
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "//tensorflow_io/ffmpeg:ffmpeg_2.8_ops",
    ],
)

cc_binary(
    name = "python/ops/libtensorflow_io_libav_9.20.so",
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "//tensorflow_io/ffmpeg:libav_9.20_ops",
    ],
)
