licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_lmdb_ops.so',
    srcs = [
        "kernels/lmdb_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@lmdb//:lmdb",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

py_library(
    name = "lmdb_ops_py",
    srcs = ([
        "python/ops/lmdb_dataset_ops.py",
    ]),
    data = [
        ":python/ops/_lmdb_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "lmdb_dataset_op_test",
    srcs = [
        "python/kernel_tests/lmdb_dataset_op_test.py"
    ],
    main = "python/kernel_tests/lmdb_dataset_op_test.py",
    deps = [
        ":lmdb_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "lmdb_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":lmdb_ops_py"
    ],
    srcs_version = "PY2AND3",
)
