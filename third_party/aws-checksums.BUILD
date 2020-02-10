# Description:
#   AWS CheckSums

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "aws-checksums",
    srcs = glob([
        "include/aws/checksums/*.h",
        "include/aws/checksums/private/*.h",
        "source/*.c",
    ]) + select({
        "@bazel_tools//src/conditions:windows": glob([
            "source/visualc/*.c",
        ]),
        "//conditions:default": glob([
            "source/intel/*.c",
        ]),
    }),
    hdrs = [],
    defines = [],
    includes = [
        "include",
    ],
    deps = [],
)
