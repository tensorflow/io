# Description:
# BigQueryReader implementation

licenses(["notice"])  # Apache 2.0


cc_library(
    name = "bigquery_reader_ops_op_lib",
    srcs = [
        "ops/bigquery_reader_ops.cc",
    ],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "gcs_config_ops_op_lib",
    srcs = [
        "ops/gcs_config_ops.cc",
    ],  
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],              
)   

