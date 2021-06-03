package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_binary(
    name = "python/ops/libtensorflow_io.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:arrow_ops",
        "//tensorflow_io/core:bigquery_ops",
        "//tensorflow_io/core:audio_video_ops",
        "//tensorflow_io/core:avro_ops",
        "//tensorflow_io/core:orc_ops",
        "//tensorflow_io/core:cpuinfo",
        "//tensorflow_io/core:file_ops",
        "//tensorflow_io/core:grpc_ops",
        "//tensorflow_io/core:hdf5_ops",
        "//tensorflow_io/core:image_ops",
        "//tensorflow_io/core:json_ops",
        "//tensorflow_io/core:kafka_ops",
        "//tensorflow_io/core:kinesis_ops",
        "//tensorflow_io/core:lmdb_ops",
        "//tensorflow_io/core:numpy_ops",
        "//tensorflow_io/core:parquet_ops",
        "//tensorflow_io/core:pcap_ops",
        "//tensorflow_io/core:pulsar_ops",
        "//tensorflow_io/core:obj_ops",
        "//tensorflow_io/core:operation_ops",
        "//tensorflow_io/core:pubsub_ops",
        "//tensorflow_io/core:serialization_ops",
        "//tensorflow_io/core:sql_ops",
        "//tensorflow_io/core:text_ops",
        "//tensorflow_io/core:ignite_ops",
        "//tensorflow_io/core:mongodb_ops",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "//tensorflow_io/core:core_ops",
            "//tensorflow_io/core:elasticsearch_ops",
            "//tensorflow_io/core:genome_ops",
            "//tensorflow_io/core:optimization",
            "//tensorflow_io/core:oss_ops",
            "//tensorflow_io/core/kernels/gsmemcachedfs:gs_memcached_file_system",
        ],
    }) + select({
        "//tensorflow_io/core:static_build_on": [
            "//tensorflow_io/core/plugins:plugins",
        ],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "python/ops/libtensorflow_io_plugins.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = select({
        "//tensorflow_io/core:static_build_on": [],
        "//conditions:default": [
            "//tensorflow_io/core/plugins:plugins",
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
    name = "python/ops/libtensorflow_io_ffmpeg_4.2.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:ffmpeg_4.2_ops",
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

py_library(
    name = "init",
    srcs = [
        "__init__.py",
    ],
)
