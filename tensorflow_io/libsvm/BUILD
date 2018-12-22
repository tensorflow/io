licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_libsvm_ops.so",
    srcs = [
        "kernels/decode_libsvm_op.cc",
        "ops/libsvm_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

py_library(
    name = "libsvm_ops_py",
    srcs = [
        "python/ops/libsvm_dataset_ops.py",
    ],
    data = [
        ":python/ops/_libsvm_ops.so",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "decode_libsvm_op_test",
    srcs = [
	"python/kernel_tests/decode_libsvm_op_test.py"
    ],
    main = "python/kernel_tests/decode_libsvm_op_test.py",
    deps = [
        ":libsvm_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "libsvm_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":libsvm_ops_py"
    ],
    srcs_version = "PY2AND3",
)
