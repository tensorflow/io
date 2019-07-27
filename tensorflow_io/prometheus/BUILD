licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_io:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "prometheus_ops",
    srcs = [
        #"//tensorflow_io/prometheus/go:prometheus.a",
        #"//tensorflow_io/prometheus/go:prometheus.h",
        "kernels/prometheus_input.cc",
        "ops/prometheus_ops.cc",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/prometheus/go:prometheus.cc",
    ],
)
