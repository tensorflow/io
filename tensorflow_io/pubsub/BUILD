licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "pubsub_ops",
    srcs = [
        "kernels/pubsub_dataset_ops.cc",
        "ops/dataset_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googleapis//google/pubsub/v1:pubsub_cc_grpc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
