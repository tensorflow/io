licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_kafka_ops.so',
    srcs = [
        "kernels/kafka_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@kafka//:kafka",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

py_library(
    name = "kafka_ops_py",
    srcs = ([
        "python/ops/kafka_dataset_ops.py",
    ]),
    data = [
        ":python/ops/_kafka_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "kafka_py_test",
    srcs = [
        "python/kernel_tests/kafka_test.py"
    ],
    main = "python/kernel_tests/kafka_test.py",
    deps = [
        ":kafka_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "kafka_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":kafka_ops_py"
    ],
    srcs_version = "PY2AND3",
)
