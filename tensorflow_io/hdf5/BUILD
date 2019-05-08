licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hdf5_ops",
    srcs = [
        "kernels/hdf5_input.cc",
        "ops/hdf5_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@hdf5",
    ],
)
