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
    copts = select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "-Wno-register",
            "-Wno-error",
        ],
    }),
    includes = [
        "include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@dcmtk",
        "@openjpeg",
    ],
)
