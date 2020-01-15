package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "oss_ops",
    srcs = [
        "kernels/ossfs/oss_file_system.cc",
        "kernels/ossfs/oss_file_system.h",
        "ops/ossfs_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@aliyun_oss_c_sdk",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
