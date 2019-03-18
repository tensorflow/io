licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_lmdb_ops.so",
    srcs = [
        "kernels/lmdb_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "@lmdb",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
