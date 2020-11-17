licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "az",
    srcs = [
        "az_file_system.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/plugins:plugins_header",
        "@com_github_azure_azure_storage_cpplite//:azure",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
