licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "image_ops",
    srcs = [
        "kernels/font_opensans_regular.h",
        "kernels/gif_dataset_ops.cc",
        "kernels/image_kernels.cc",
        "kernels/image_tiff_kernels.cc",
        "kernels/tiff_dataset_ops.cc",
        "kernels/webp_dataset_ops.cc",
        "ops/dataset_ops.cc",
        "ops/image_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@freetype",
        "@giflib",
        "@libtiff",
        "@libwebp",
    ],
)
