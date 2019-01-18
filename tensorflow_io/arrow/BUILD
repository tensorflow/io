licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_arrow_ops.so',
    srcs = [
        "kernels/arrow_dataset_ops.cc",
        "kernels/arrow_stream_client.h",
        "kernels/arrow_stream_client_unix.cc",
        "ops/dataset_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@arrow//:arrow",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

py_library(
    name = "arrow_ops_py",
    srcs = [
        "python/ops/arrow_dataset_ops.py",
    ],
    data = [
        ":python/ops/_arrow_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "arrow_py_test",
    srcs = [
        "python/kernel_tests/arrow_test.py"
    ],
    main = "python/kernel_tests/arrow_test.py",
    deps = [
        ":arrow_ops_py"
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "arrow_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ],
    deps = [
        ":arrow_ops_py",
    ],
    srcs_version = "PY2AND3",
)
