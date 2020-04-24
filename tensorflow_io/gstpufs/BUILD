load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

licenses(["notice"])

cc_library(
    name = "gs_tpu_file_system",
    srcs = ["gs_tpu_file_system.cc"],
    hdrs = ["gs_tpu_file_system.h"],
    copts = tf_io_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:libtensorflow_framework",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)
