# Description:
#   JPEG2000 codec for DCMTK based on openjpeg

licenses(["notice"])  # Apache 2.0 license

exports_files(["LICENSE"])

cc_library(
    name = "fmjpeg2koj",
    srcs = glob([
        "include/fmjpeg2k/*.h",
        "*.cc",
        "*.cpp",
    ]),
    includes = [
        "include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@dcmtk",
        "@openjpeg",
    ],
)
