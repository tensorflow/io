licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "ignite_ops",
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
        "kernels/ggfs/ggfs.cc",
        "kernels/ggfs/ggfs.h",
        "kernels/ggfs/ggfs_client.cc",
        "kernels/ggfs/ggfs_client.h",
        "kernels/ggfs/ggfs_random_access_file.cc",
        "kernels/ggfs/ggfs_random_access_file.h",
        "kernels/ggfs/ggfs_writable_file.cc",
        "kernels/ggfs/ggfs_writable_file.h",
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
        "ops/ggfs_ops.cc",
        "ops/igfs_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@boringssl//:ssl",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
