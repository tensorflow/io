licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_hadoop_ops.so',
    srcs = [
        "kernels/hadoop_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)
