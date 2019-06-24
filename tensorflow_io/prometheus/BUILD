licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "prometheus_ops",
    srcs = [
        #"//tensorflow_io/prometheus/go:prometheus.a",
        #"//tensorflow_io/prometheus/go:prometheus.h",
        "kernels/prometheus_input.cc",
        "ops/prometheus_ops.cc",
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
        "//tensorflow_io/prometheus/go:prometheus.cc",
    ],
)
