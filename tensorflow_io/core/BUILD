licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)
load("@io_bazel_rules_go//go:def.bzl", "go_binary")

cc_library(
    name = "cpuinfo",
    srcs = [
        "kernels/cpu_check.cc",
        "kernels/cpu_info.cc",
        "kernels/cpu_info.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sequence_ops",
    srcs = [
        "kernels/sequence_ops.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dataset_ops",
    srcs = [
        "kernels/io_interface.h",
        "kernels/io_stream.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "output_ops",
    srcs = [
        "kernels/output_ops.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "core_ops",
    srcs = [
        "kernels/archive_kernels.cc",
        "kernels/http_kernels.cc",
        "ops/core_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        ":dataset_ops",
        "@curl",
        "@libarchive",
    ],
    alwayslink = 1,
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
        "@com_github_prometheus_client_model//go:go_default_library",
        "@com_github_prometheus_common//model:go_default_library",
        "@com_github_prometheus_prom2json//:go_default_library",
    ],
)

cc_library(
    name = "go_ops",
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:golang_ops.cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lmdb_ops",
    srcs = [
        "kernels/lmdb_kernels.cc",
        "ops/lmdb_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@lmdb",
    ],
    alwayslink = 1,
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
    alwayslink = 1,
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
    alwayslink = 1,
)

cc_library(
    name = "image_ops",
    srcs = [
        "ops/image_ops.cc",
        "kernels/image_bmp_kernels.cc",
        "kernels/image_font_kernels.cc",
        "kernels/image_jpeg_kernels.cc",
        "kernels/image_hdr_kernels.cc",
        "kernels/image_openexr_kernels.cc",
        "kernels/image_pnm_kernels.cc",
        "kernels/image_webp_kernels.cc",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "kernels/image_dicom_kernels.cc",
            "kernels/image_tiff_kernels.cc",
        ],
    }),
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//third_party:font",
        "@easyexif",
        "@freetype",
        "@libwebp",
        "@openexr",
        "@stb",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "@dcmtk",
            "@libtiff",
        ],
    }),
    alwayslink = 1,
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
        "@flac",
        "@vorbis",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ffmpeg_3.4_ops",
    srcs = [
        "kernels/ffmpeg_kernels.cc",
        "kernels/ffmpeg_kernels_deprecated.cc",
        "ops/ffmpeg_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_3_4//:ffmpeg",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ffmpeg_2.8_ops",
    srcs = [
        "kernels/ffmpeg_kernels.cc",
        "kernels/ffmpeg_kernels_deprecated.cc",
        "ops/ffmpeg_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_2_8//:ffmpeg",
    ],
    alwayslink = 1,
)

cc_library(
    name = "avro_ops",
    srcs = [
        "kernels/avro_kernels.cc",
        "ops/avro_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@avro",
    ],
    alwayslink = 1,
)

cc_library(
    name = "text_ops",
    srcs = [
        "kernels/csv_kernels.cc",
        "kernels/libsvm_kernels.cc",
        "kernels/text_kernels.cc",
        "kernels/text_re2.cc",
        "kernels/text_sequence.cc",
        "ops/libsvm_ops.cc",
        "ops/text_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/arrow:arrow_ops",
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core:output_ops",
        "//tensorflow_io/core:sequence_ops",
        "@com_googlesource_code_re2//:re2",
    ],
    alwayslink = 1,
)

cc_library(
    name = "genome_ops",
    srcs = [
        "kernels/genome_fastq_kernels.cc",
        "ops/genome_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@nucleus//:fastq_reader",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hdf5_ops",
    srcs = [
        "kernels/hdf5_kernels.cc",
        "ops/hdf5_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@hdf5",
    ],
    alwayslink = 1,
)

cc_library(
    name = "parquet_ops",
    srcs = [
        "kernels/parquet_kernels.cc",
        "ops/parquet_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/arrow:arrow_ops",
        "//tensorflow_io/core:dataset_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pubsub_ops",
    srcs = [
        "kernels/pubsub_kernels.cc",
        "ops/pubsub_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googleapis//google/pubsub/v1:pubsub_cc_grpc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "kinesis_ops",
    srcs = [
        "kernels/kinesis_kernels.cc",
        "ops/kinesis_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@aws",
        "@curl",
    ],
    alwayslink = 1,
)

cc_library(
    name = "file_ops",
    srcs = [
        "kernels/file_kernels.cc",
        "ops/file_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "serialization_ops",
    srcs = [
        "kernels/serialization_kernels.cc",
        "ops/serialization_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@avro",
        "@rapidjson",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_ops",
    srcs = [
        "kernels/grpc_kernels.cc",
        "ops/grpc_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core/grpc:endpoint_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
    ],
    alwayslink = 1,
)

cc_library(
    name = "json_ops",
    srcs = [
        "kernels/json_kernels.cc",
        "ops/json_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/arrow:arrow_ops",
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "numpy_ops",
    srcs = [
        "kernels/numpy_kernels.cc",
        "ops/numpy_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@zlib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "kafka_ops",
    srcs = [
        "kernels/kafka_kernels.cc",
        "kernels/kafka_kernels_deprecated.cc",
        "ops/kafka_ops.cc",
        "ops/kafka_ops_deprecated.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core:sequence_ops",
        "@avro",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@kafka",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "azfs_ops",
    srcs = [
        "kernels/azfs_kernels.cc",
        "kernels/azfs_kernels.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@com_github_azure_azure_storage_cpplite//:azure",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

# bazel test \
#   --action_env=TF_AZURE_USE_DEV_STORAGE=1 \
#   //tensorflow_io/azure:azfs_ops_test
cc_test(
    name = "azfs_ops_test",
    srcs = [
        "kernels/azfs_kernels_test.cc",
    ],
    deps = [
        ":azfs_ops",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "python/ops/libtensorflow_io.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:azfs_ops",
        "//tensorflow_io/core:image_ops",
        "//tensorflow_io/core:kafka_ops",
        "//tensorflow_io/core:lmdb_ops",
        "//tensorflow_io/core:numpy_ops",
        "//tensorflow_io/core:serialization_ops",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "//tensorflow_io/arrow:arrow_ops",
            "//tensorflow_io/bigquery:bigquery_ops",
            "//tensorflow_io/bigtable:bigtable_ops",
            "//tensorflow_io/core:audio_ops",
            "//tensorflow_io/core:avro_ops",
            "//tensorflow_io/core:core_ops",
            "//tensorflow_io/core:cpuinfo",
            "//tensorflow_io/core:file_ops",
            "//tensorflow_io/core:genome_ops",
            "//tensorflow_io/core:grpc_ops",
            "//tensorflow_io/core:hdf5_ops",
            "//tensorflow_io/core:json_ops",
            "//tensorflow_io/core:kinesis_ops",
            "//tensorflow_io/core:parquet_ops",
            "//tensorflow_io/core:pcap_ops",
            "//tensorflow_io/core:pubsub_ops",
            "//tensorflow_io/core:text_ops",
            "//tensorflow_io/core/oss:oss_ops",
            "//tensorflow_io/gcs:gcs_config_ops",
            "//tensorflow_io/ignite:ignite_ops",
            "@libarchive",
            "@xz//:lzma",
        ],
    }),
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
