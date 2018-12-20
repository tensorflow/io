licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_ignite_ops.so',
    srcs = [
        "ops/dataset_ops.cc",
        "ops/igfs_ops.cc",
        "kernels/igfs/igfs.cc",
        "kernels/igfs/igfs.h",
        "kernels/igfs/igfs_client.cc",
        "kernels/igfs/igfs_client.h",
        "kernels/igfs/igfs_extended_tcp_client.cc",
        "kernels/igfs/igfs_extended_tcp_client.h",
        "kernels/igfs/igfs_messages.cc",
        "kernels/igfs/igfs_messages.h",
        "kernels/igfs/igfs_random_access_file.cc",
        "kernels/igfs/igfs_random_access_file.h",
        "kernels/igfs/igfs_writable_file.cc",
        "kernels/igfs/igfs_writable_file.h",
        "kernels/dataset/ignite_binary_object_parser.cc",
        "kernels/dataset/ignite_binary_object_parser.h",
        "kernels/dataset/ignite_dataset.cc",
        "kernels/dataset/ignite_dataset.h",
        "kernels/dataset/ignite_dataset_iterator.cc",
        "kernels/dataset/ignite_dataset_iterator.h",
        "kernels/dataset/ignite_dataset_ops.cc",
        "kernels/client/ignite_client.h",
        "kernels/client/ignite_byte_swapper.h",
        "kernels/client/ignite_plain_client.h",
        "kernels/client/ignite_ssl_wrapper.h",
        "kernels/client/ignite_ssl_wrapper.cc",
        "kernels/client/ignite_plain_client_unix.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@boringssl//:ssl",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

py_library(
    name = "dataset_ops_py",
    srcs = [
        "python/ops/ignite_dataset_ops.py",
    ],
    data = [
        ":python/ops/_ignite_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "igfs_ops_py",
    srcs = [
        "python/ops/igfs_ops.py",
    ],
    data = [
        ":python/ops/_ignite_ops.so"
    ],
    srcs_version = "PY2AND3",
)

# The Apache Ignite servers have to setup before the test and tear down
# after the test manually. The docker engine has to be installed.
#
# To setup Apache Ignite servers:
# $ bash ./python/tests/start_ignite.sh
#
# To tear down Apache Ignite servers:
# $ bash ./python/tests/stop_ignite.sh
py_test(
    name = "ignite_py_test",
    srcs = [
        "python/tests/ignite_dataset_test.py",
    ],
    main = "python/tests/ignite_dataset_test.py",
    deps = [
        ":ignite_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "igfs_py_test",
    srcs = [
        "python/tests/igfs_test.py",
    ],
    main = "python/tests/igfs_test.py",
    deps = [
        ":ignite_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "ignite_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ],
    deps = [
        ":dataset_ops_py",
        ":igfs_ops_py",
    ],
    srcs_version = "PY2AND3",
)
