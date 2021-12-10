licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "az",
    srcs = [
        "az_filesystem.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/filesystems:filesystem_plugins_header",
        "@com_github_azure_azure_sdk_for_cpp//:azure",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
