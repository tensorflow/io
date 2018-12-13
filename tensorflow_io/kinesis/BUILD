licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_kinesis_ops.so',
    srcs = [
        "kernels/kinesis_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@curl//:curl",
        "@aws//:aws",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

py_library(
    name = "kinesis_ops_py",
    srcs = ([
        "python/ops/kinesis_dataset_ops.py",
    ]),
    data = [
        ":python/ops/_kinesis_ops.so"
    ],
    srcs_version = "PY2AND3",
)

# Stub test to make sure kinesis ops is loaded.
py_test(
    name = "kinesis_stub_test",
    srcs = [
        "python/kernel_tests/kinesis_stub_test.py"
    ],
    main = "python/kernel_tests/kinesis_stub_test.py",
    deps = [
        ":kinesis_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "kinesis_py_test",
    srcs = [
        "python/kernel_tests/kinesis_test.py"
    ],
    main = "python/kernel_tests/kinesis_test.py",
    deps = [
        ":kinesis_ops_py",
    ],
    srcs_version = "PY2AND3",
    tags = ["manual"],
)

py_library(
    name = "kinesis_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":kinesis_ops_py"
    ],
    srcs_version = "PY2AND3",
)
