licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)
load("@io_bazel_rules_go//go:def.bzl", "go_binary")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

bool_flag(
    name = "static_build",
    build_setting_default = False,
)

config_setting(
    name = "static_build_on",
    flag_values = {
        ":static_build": "True",
    },
)

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
    name = "operation_ops",
    srcs = [
        "kernels/operation_kernels.cc",
        "ops/operation_ops.cc",
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
    name = "optimization",
    srcs = [
        "kernels/io_optimization.cc",
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
        "kernels/io_kernel.h",
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
    name = "arrow_ops",
    srcs = [
        "kernels/arrow/arrow_dataset_ops.cc",
        "kernels/arrow/arrow_kernels.cc",
        "kernels/arrow/arrow_kernels.h",
        "kernels/arrow/arrow_stream_client.h",
        "ops/arrow_ops.cc",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "kernels/arrow/arrow_stream_client_windows.cc",
        ],
        "//conditions:default": [
            "kernels/arrow/arrow_stream_client_unix.cc",
        ],
    }),
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        ":arrow_util",
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
    ],
    alwayslink = 1,
)

cc_library(
    name = "arrow_util",
    srcs = ["kernels/arrow/arrow_util.cc"],
    hdrs = ["kernels/arrow/arrow_util.h"],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
    ],
    alwayslink = 1,
)

# Have to put test op into prod binary because of https://github.com/grpc/grpc/issues/20034
# Otherwise whenever 2 dynamically built libraries that are using grpc loaded
# using _load_library, only first one will work
cc_library(
    name = "bigquery_ops",
    srcs = [
        "kernels/bigquery/bigquery_dataset_op.cc",
        "kernels/bigquery/bigquery_kernels.cc",
        "kernels/tests/bigquery_test_client_op.cc",
        "ops/bigquery_ops.cc",
        "ops/bigquery_test_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        ":bigquery_lib_cc",
        "@arrow",
        "@avro",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googleapis//google/cloud/bigquery/storage/v1beta1:storage_cc_grpc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

# A library for use in the bigquery kernels.
cc_library(
    name = "bigquery_lib_cc",
    srcs = ["kernels/bigquery/bigquery_lib.cc"],
    hdrs = ["kernels/bigquery/bigquery_lib.h"],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:arrow_util",
        "@arrow",
        "@avro",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:variant",
        "@com_google_googleapis//google/cloud/bigquery/storage/v1beta1:storage_cc_grpc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "core_ops",
    srcs = [
        "kernels/archive_kernels.cc",
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
    name = "elasticsearch_ops",
    srcs = [
        "kernels/elasticsearch_kernels.cc",
        "ops/elasticsearch_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core:sequence_ops",
        "@curl",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@rapidjson",
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
        "kernels/image_avif_kernels.cc",
        "kernels/image_bmp_kernels.cc",
        "kernels/image_dicom_kernels.cc",
        "kernels/image_font_kernels.cc",
        "kernels/image_gif_kernels.cc",
        "kernels/image_hdr_kernels.cc",
        "kernels/image_jpeg2k_kernels.cc",
        "kernels/image_jpeg_kernels.cc",
        "kernels/image_nv12_kernels.cc",
        "kernels/image_openexr_kernels.cc",
        "kernels/image_pnm_kernels.cc",
        "kernels/image_tiff_kernels.cc",
        "kernels/image_webp_kernels.cc",
        "kernels/image_yuy2_kernels.cc",
        "ops/image_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//third_party:font",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:variant",
        "@easyexif",
        "@fmjpeg2koj",
        "@freetype",
        "@gif",
        "@libavif",
        "@libgeotiff",
        "@libtiff",
        "@libwebp",
        "@libyuv",
        "@openexr",
        "@openjpeg",
        "@stb",
    ],
    alwayslink = 1,
)

exports_files([
    "swift/audio.swift",
    "swift/video.swift",
])

cc_library(
    name = "audio_video_ops",
    srcs = [
        "kernels/audio_kernels.cc",
        "kernels/audio_kernels.h",
        "kernels/audio_video_flac_kernels.cc",
        "kernels/audio_video_mp3_kernels.cc",
        "kernels/audio_video_mp4_kernels.cc",
        "kernels/audio_video_ogg_kernels.cc",
        "kernels/audio_video_wav_kernels.cc",
        "kernels/video_kernels.cc",
        "kernels/video_kernels.h",
        "ops/audio_ops.cc",
        "ops/video_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@flac",
        "@minimp3",
        "@speexdsp",
        "@minimp4",
        "@vorbis",
        "//tensorflow_io/core:dataset_ops",
    ] + select({
        "@bazel_tools//src/conditions:darwin": [
            "//tools/build/swift:audio_video_swift",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "ffmpeg_4.2_ops",
    srcs = [
        "kernels/audio_ffmpeg_kernels.cc",
        "kernels/ffmpeg_kernels.cc",
        "kernels/ffmpeg_kernels_deprecated.cc",
        "ops/ffmpeg_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@ffmpeg_4_2//:ffmpeg",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ffmpeg_3.4_ops",
    srcs = [
        "kernels/audio_ffmpeg_kernels.cc",
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
        "kernels/avro/avro_record_dataset_kernels.cc",
        "kernels/avro/parse_avro_kernels.cc",
        "kernels/avro_kernels.cc",
        "ops/avro_ops.cc",
    ],
    hdrs = [
        "kernels/avro/avro_record_dataset_kernels.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core/kernels/avro/utils:avro_utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "orc_ops",
    srcs = [
        "kernels/orc/orc_kernels.cc",
        "ops/orc_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@liborc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
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
        ":arrow_ops",
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
        ":arrow_ops",
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
        "@com_google_absl//absl/time",
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
        "@aws-sdk-cpp//:kinesis",
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
        ":arrow_ops",
        "//tensorflow_io/core:dataset_ops",
        "@arrow",
        "@rapidjson",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@kafka",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "oss_ops",
    srcs = [
        "kernels/oss/oss_file_system.cc",
        "kernels/oss/oss_file_system.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@aliyun_oss_c_sdk",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sql_ops",
    srcs = [
        "kernels/sql_kernels.cc",
        "ops/sql_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@postgresql",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ignite_ops",
    srcs = [
        "kernels/ignite/client/ignite_byte_swapper.h",
        "kernels/ignite/client/ignite_client.h",
        "kernels/ignite/client/ignite_plain_client.h",
        "kernels/ignite/client/ignite_ssl_wrapper.cc",
        "kernels/ignite/client/ignite_ssl_wrapper.h",
        "kernels/ignite/dataset/ignite_binary_object_parser.cc",
        "kernels/ignite/dataset/ignite_binary_object_parser.h",
        "kernels/ignite/dataset/ignite_dataset.cc",
        "kernels/ignite/dataset/ignite_dataset.h",
        "kernels/ignite/dataset/ignite_dataset_iterator.cc",
        "kernels/ignite/dataset/ignite_dataset_iterator.h",
        "kernels/ignite/dataset/ignite_dataset_ops.cc",
        "kernels/ignite/ggfs/ggfs.cc",
        "kernels/ignite/ggfs/ggfs.h",
        "kernels/ignite/ggfs/ggfs_client.cc",
        "kernels/ignite/ggfs/ggfs_client.h",
        "kernels/ignite/ggfs/ggfs_random_access_file.cc",
        "kernels/ignite/ggfs/ggfs_random_access_file.h",
        "kernels/ignite/ggfs/ggfs_writable_file.cc",
        "kernels/ignite/ggfs/ggfs_writable_file.h",
        "kernels/ignite/igfs/igfs.cc",
        "kernels/ignite/igfs/igfs.h",
        "kernels/ignite/igfs/igfs_client.cc",
        "kernels/ignite/igfs/igfs_client.h",
        "kernels/ignite/igfs/igfs_extended_tcp_client.cc",
        "kernels/ignite/igfs/igfs_extended_tcp_client.h",
        "kernels/ignite/igfs/igfs_messages.cc",
        "kernels/ignite/igfs/igfs_messages.h",
        "kernels/ignite/igfs/igfs_random_access_file.cc",
        "kernels/ignite/igfs/igfs_random_access_file.h",
        "kernels/ignite/igfs/igfs_writable_file.cc",
        "kernels/ignite/igfs/igfs_writable_file.h",
        "ops/ignite_ops.cc",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "kernels/ignite/client/ignite_plain_client_windows.cc",
        ],
        "//conditions:default": [
            "kernels/ignite/client/ignite_plain_client_unix.cc",
        ],
    }),
    copts = tf_io_copts(),
    defines = [
        "WIN32_LEAN_AND_MEAN",
    ],
    linkstatic = True,
    deps = [
        "@boringssl//:ssl",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pulsar_ops",
    srcs = [
        "kernels/pulsar_kernel.cc",
        "ops/pulsar_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@pulsar",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mongodb_ops",
    srcs = [
        "kernels/mongodb_kernels.cc",
        "ops/mongodb_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@libmongoc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "obj_ops",
    srcs = [
        "kernels/obj_kernels.cc",
        "ops/obj_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@tinyobjloader",
    ],
    alwayslink = 1,
)

cc_library(
    name = "filesystem_ops",
    srcs = [
        "kernels/filesystem_kernels.cc",
        "ops/filesystem_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
