licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "oss",
    srcs = [
        "oss_filesystem.cc",
        "oss_filesystem.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/filesystems:filesystem_plugins_header",
        "@aliyun_oss_c_sdk",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
