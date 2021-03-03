licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "gs",
    srcs = [
        "cleanup.h",
        "expiring_lru_cache.h",
        "gcs_filesystem.cc",
        "gcs_helper.cc",
        "gcs_helper.h",
        "ram_file_block_cache.cc",
        "ram_file_block_cache.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/plugins:plugins_header",
        "@com_github_googleapis_google_cloud_cpp//:storage_client",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:variant",
    ],
    alwayslink = 1,
)
