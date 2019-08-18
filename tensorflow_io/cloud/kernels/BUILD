# Description:
# BigQueryReader implementation

package(
    default_visibility = ["//tensorflow_io/cloud:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "bigquery_reader_ops",
    srcs = ["bigquery_reader_ops.cc"],
    linkstatic = True,
    deps = [
        ":bigquery_table_accessor",
        ":bigquery_table_partition_proto_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "bigquery_table_accessor",
    srcs = ["bigquery_table_accessor.cc"],
    hdrs = ["bigquery_table_accessor.h"],
    linkstatic = True,
    deps = [
        ":bigquery_table_partition_proto_cc",
        "@curl",
        "@jsoncpp_git//:jsoncpp",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_proto_library(
    name = "bigquery_table_partition_proto_cc",
    deps = [":bigquery_table_partition_proto"],
)

proto_library(
    name = "bigquery_table_partition_proto",
    srcs = ["bigquery_table_partition.proto"],
)

cc_library(
    name = "gcs_config_ops",
    srcs = ["gcs_config_ops.cc"],
    linkstatic = True,
    deps = [
        "@curl",
        "@jsoncpp_git//:jsoncpp",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
