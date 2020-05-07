load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

licenses(["notice"])

cc_library(
    name = "memcached_dao_interfaces",
    hdrs = [
        "memcached_dao_interface.h",
    ],
    copts = tf_io_copts(),
    visibility = ["//visibility:public"],
    deps = [
    ] + select({
        "@bazel_tools//src/conditions:darwin": [],
        "//conditions:default": [
            "@libmemcached",
        ],
    }),
)

cc_library(
    name = "gs_tpu_file_system",
    srcs = ["gs_tpu_file_system.cc"],
    hdrs = ["gs_tpu_file_system.h"],
    copts = tf_io_copts(),
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":memcached_dao_interfaces",
        "@com_google_absl//absl/memory",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
