licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mnist_ops",
    srcs = [
        "kernels/mnist_input.cc",
        "ops/mnist_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
    ],
)
