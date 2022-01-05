licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "dfs",
    srcs = [
        "dfs_filesystem.cc",
        "dfs_utils.cc",
        "dfs_utils.h",
    ],
    copts = tf_io_copts(),
    linkopts = [
        "-ldaos",
        "-luuid",
        "-ldfs",
        "-lduns",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/filesystems:filesystem_plugins_header",
        "@daos",
    ],
    alwayslink = 1,
)
