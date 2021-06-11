licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "http",
    srcs = [
        "http_filesystem.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/filesystems:filesystem_plugins_header",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@curl",
    ],
    alwayslink = 1,
)
