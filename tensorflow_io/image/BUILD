licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_io:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_binary(
    name = "python/ops/_image_ops.so",
    srcs = [
        "kernels/font_opensans_regular.h",
        "kernels/gif_dataset_ops.cc",
        "kernels/image_kernels.cc",
        "kernels/tiff_dataset_ops.cc",
        "kernels/webp_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkshared = 1,
    deps = [
        "@freetype",
        "@giflib",
        "@libtiff",
        "@libwebp",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
