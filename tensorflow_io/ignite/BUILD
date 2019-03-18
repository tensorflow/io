licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_ignite_ops.so",
    srcs = [
        "kernels/client/ignite_byte_swapper.h",
        "kernels/client/ignite_client.h",
        "kernels/client/ignite_plain_client.h",
        "kernels/client/ignite_plain_client_unix.cc",
        "kernels/client/ignite_ssl_wrapper.cc",
        "kernels/client/ignite_ssl_wrapper.h",
        "kernels/dataset/ignite_binary_object_parser.cc",
        "kernels/dataset/ignite_binary_object_parser.h",
        "kernels/dataset/ignite_dataset.cc",
        "kernels/dataset/ignite_dataset.h",
        "kernels/dataset/ignite_dataset_iterator.cc",
        "kernels/dataset/ignite_dataset_iterator.h",
        "kernels/dataset/ignite_dataset_ops.cc",
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
        "ops/dataset_ops.cc",
        "ops/igfs_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "@boringssl//:ssl",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
