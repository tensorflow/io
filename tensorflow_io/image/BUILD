licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_image_ops.so',
    srcs = [
        "kernels/image_ops.h",
        "kernels/image_ops.cc",
        "kernels/webp_dataset_ops.cc",
        "kernels/tiff_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    includes = [
        ".",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
	"@libwebp//:libwebp",
	"@libtiff//:libtiff",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)
