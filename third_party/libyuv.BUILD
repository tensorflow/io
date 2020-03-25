# Description:
#   libyuv library from Chromium

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "libyuv",
    srcs = glob([
        "include/libyuv/*.h",
        "source/row_*.cc",
        "source/scale_*.cc",
    ]) + [
        "source/convert_argb.cc",
        "source/convert_from_argb.cc",
        "source/cpu_id.cc",
        "source/planar_functions.cc",
    ],
    includes = [
        "include",
    ],
    visibility = ["//visibility:public"],
)
