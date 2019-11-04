licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)
load("@io_bazel_rules_go//go:def.bzl", "go_binary")

cc_library(
    name = "sequence_ops",
    srcs = [
        "kernels/sequence_ops.h",
    ],
    copts = tf_io_copts(),
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
        "kernels/io_interface.h",
        "kernels/io_stream.h",
    ],
    copts = tf_io_copts(),
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
    name = "output_ops",
    srcs = [
        "kernels/output_ops.h",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "core_ops",
    srcs = [
        "kernels/archive_kernels.cc",
        "ops/core_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        ":dataset_ops",
    ],
)

go_binary(
    name = "golang_ops",
    srcs = ["go/prometheus.go"],
    cgo = True,
    linkmode = "c-archive",
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_prometheus_client_golang//api:go_default_library",
        "@com_github_prometheus_client_golang//api/prometheus/v1:go_default_library",
        "@com_github_prometheus_common//model:go_default_library",
    ],
)

cc_library(
    name = "go_ops",
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:golang_ops.cc",
    ],
)

cc_library(
    name = "prometheus_ops",
    srcs = [
        "kernels/prometheus_kernels.cc",
        "ops/prometheus_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core:go_ops",
    ],
)

cc_library(
    name = "pcap_ops",
    srcs = [
        "kernels/pcap_kernels.cc",
        "ops/pcap_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "image_ops",
    srcs = [
        "kernels/font_opensans_regular.h",
        "kernels/image_bmp_kernels.cc",
        "kernels/image_kernels.cc",
        "kernels/image_tiff_kernels.cc",
        "kernels/image_webp_kernels.cc",
        "ops/image_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@freetype",
        "@giflib",
        "@libtiff",
        "@libwebp",
    ],
)

cc_library(
    name = "audio_ops",
    srcs = [
        "kernels/audio_kernels.cc",
        "ops/audio_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
    ],
)

cc_library(
    name = "ffmpeg_3.4_ops",
    srcs = [
        "kernels/ffmpeg_kernels.cc",
        "ops/ffmpeg_ops.cc",
    ],
    copts = tf_io_copts(),
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
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_2_8//:ffmpeg",
    ],
)

cc_binary(
    name = "python/ops/libtensorflow_io.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/arrow:arrow_ops",
        "//tensorflow_io/avro:avro_ops",
        "//tensorflow_io/azure:azfs_ops",
        "//tensorflow_io/bigquery:bigquery_ops",
        "//tensorflow_io/bigtable:bigtable_ops",
        "//tensorflow_io/core:audio_ops",
        "//tensorflow_io/core:core_ops",
        "//tensorflow_io/core:image_ops",
        "//tensorflow_io/core:pcap_ops",
        "//tensorflow_io/dicom:dicom_ops",
        "//tensorflow_io/gcs:gcs_config_ops",
        "//tensorflow_io/genome:genome_ops",
        "//tensorflow_io/grpc:grpc_ops",
        "//tensorflow_io/hdf5:hdf5_ops",
        "//tensorflow_io/ignite:ignite_ops",
        "//tensorflow_io/json:json_ops",
        "//tensorflow_io/kafka:kafka_ops",
        "//tensorflow_io/kinesis:kinesis_ops",
        "//tensorflow_io/lmdb:lmdb_ops",
        "//tensorflow_io/mnist:mnist_ops",
        "//tensorflow_io/oss:oss_ops",
        "//tensorflow_io/parquet:parquet_ops",
        "//tensorflow_io/pubsub:pubsub_ops",
        "//tensorflow_io/text:text_ops",
        "@libarchive",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@xz//:lzma",
    ],
)

cc_binary(
    name = "python/ops/libtensorflow_io_golang.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:prometheus_ops",
    ],
)

cc_binary(
    name = "python/ops/libtensorflow_io_ffmpeg_3.4.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:ffmpeg_3.4_ops",
    ],
)

cc_binary(
    name = "python/ops/libtensorflow_io_ffmpeg_2.8.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:ffmpeg_2.8_ops",
    ],
)
