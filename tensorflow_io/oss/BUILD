package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "python/ops/_oss_ops.so",
    srcs = [
        "kernels/ossfs/oss_file_system.cc",
        "kernels/ossfs/oss_file_system.h",
        "ops/ossfs_ops.cc",
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    deps = [
        "@aliyun_oss_c_sdk",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
