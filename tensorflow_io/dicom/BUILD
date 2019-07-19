licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_io:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "dicom_ops",
    srcs = [
        "kernels/decode_dicom_data.cc",
        "kernels/decode_dicom_image.cc",
        "ops/decode_dicom_data_op.cc",
        "ops/decode_dicom_image_op.cc",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        "@dcmtk",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)