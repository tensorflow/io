licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pcap_ops",
    srcs = [
        "kernels/pcap_input.cc",
        "ops/pcap_ops.cc",
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
