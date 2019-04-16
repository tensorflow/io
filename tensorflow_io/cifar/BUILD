licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_cifar_ops.so",
    srcs = [
        "kernels/cifar_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@libarchive",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
