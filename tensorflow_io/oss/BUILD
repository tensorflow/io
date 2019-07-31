package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_binary(
    name = "python/ops/_oss_ops.so",
    srcs = [
        "kernels/ossfs/oss_file_system.cc",
        "kernels/ossfs/oss_file_system.h",
        "ops/ossfs_ops.cc",
    ],
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "@aliyun_oss_c_sdk",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
