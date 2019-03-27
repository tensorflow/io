licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_kafka_ops.so",
    srcs = [
        "kernels/kafka_dataset_ops.cc",
        "kernels/kafka_sequence.cc",
        "ops/dataset_ops.cc",
        "ops/kafka_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "//tensorflow_io/core:sequence_ops",
        "@kafka",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
