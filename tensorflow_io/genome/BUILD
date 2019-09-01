licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)
cc_binary(
    name = "python/ops/_genome_ops.so",
    srcs = [
        "kernels/fastq.cc",
    ],
    copts = (tf_io_copts() + ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]),
    includes = [
        ".",
    ],
    linkshared = 1, 
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "//tensorflow_io/core:output_ops",
        "//tensorflow_io/core:sequence_ops",
	"@local_config_tf//:libtensorflow_framework",
	"@nucleus//nucleus/vendor:statusor",
        "@local_config_tf//:tf_header_lib",
	"@nucleus//nucleus/io:fastq_reader",
    ],
)

