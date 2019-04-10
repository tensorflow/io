licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_image_ops.so",
    srcs = [
        "kernels/gif_dataset_ops.cc",
        "kernels/tiff_dataset_ops.cc",
        "kernels/webp_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "@giflib",
        "@libtiff",
        "@libwebp",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
