licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "plugins_header",
    srcs = [
        "file_system_plugins.h",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "@local_config_tf//:stub/libtensorflow_framework.lib",
        ],
        "//conditions:default": [
            "@local_config_tf//:stub/libtensorflow_framework.so",
        ],
    }),
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:tf_c_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "plugins",
    srcs = [
        "file_system_plugins.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/plugins/az",
        "//tensorflow_io/core/plugins/hdfs",
        "//tensorflow_io/core/plugins/http",
        "//tensorflow_io/core/plugins/s3",
    ],
    alwayslink = 1,
)
