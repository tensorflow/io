licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_data",
    srcs = glob(["python/kernel_tests/testdata/*"]),
)

cc_binary(
    name = 'python/ops/_parquet_ops.so',
    srcs = [
        "kernels/parquet_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@parquet//:parquet",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

py_library(
    name = "parquet_ops_py",
    srcs = ([
        "python/ops/parquet_dataset_ops.py",
    ]),
    data = [
        ":python/ops/_parquet_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "parquet_py_test",
    srcs = [
        "python/kernel_tests/parquet_test.py"
    ],
    main = "python/kernel_tests/parquet_test.py",
    data = [
        ":test_data",
    ],
    deps = [
        ":parquet_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "parquet_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":parquet_ops_py"
    ],
    srcs_version = "PY2AND3",
)
