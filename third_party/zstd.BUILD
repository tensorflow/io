# Description:
#   Zstandard library

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

cc_library(
    name = "zstd",
    srcs = glob([
        "lib/common/*.h",
        "lib/common/*.c",
        "lib/compress/*.c",
        "lib/compress/*.h",
        "lib/decompress/*.c",
        "lib/decompress/*.h",
    ]),
    hdrs = [
        "lib/zstd.h",
    ],
    defines = [],
    includes = [
        "lib",
        "lib/common",
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
)
