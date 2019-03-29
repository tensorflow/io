# Description:
# BigQueryReader implementation

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "python/ops/_bigquery_reader_ops.so",
    srcs = [
        "//tensorflow_io/cloud/kernels:bigquery_reader_ops",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "python/ops/_gcs_config_ops.so",
    srcs = [
        "//tensorflow_io/cloud/kernels:gcs_config_ops",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

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

# TODO(henrytansetiawan): link the tensorflow_io/bigtable
py_library(
    name = "cloud_py",
    srcs = ([
        "__init__.py",
        "python/ops/bigquery_reader_ops.py",
        "python/ops/gcs_config_ops.py",
    ]),
    data = [
        ":python/ops/_bigquery_reader_ops.so",
        ":python/ops/_gcs_config_ops.so",
    ],
    srcs_version = "PY2AND3",
)

# TODO(henrytansetiawan): add bigquery_reader_ops_test
# TODO(henrytansetiawan): add gcs_config_ops_test
