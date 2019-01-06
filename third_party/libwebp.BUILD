# Description:
#   WebP codec library

licenses(["notice"])  # WebM license

exports_files(["COPYING"])

cc_library(
    name = "libwebp",
    srcs = glob([
        "src/dsp/*.c",
        "src/dsp/*.h",
        "src/utils/*.c",
        "src/utils/*.h",
        "src/dec/*.c",
        "src/dec/*.h",
        "src/demux/*.c",
        "src/demux/*.h",
        "src/enc/*.c",
        "src/enc/*.h",
    ]) + [
        "imageio/imageio_util.c",
        "imageio/webpdec.c",
        "imageio/metadata.c",
    ],
    hdrs = glob([
        "src/webp/*.h",
    ]) + [
        "imageio/webpdec.h",
        "imageio/metadata.h",
        "imageio/imageio_util.h",
    ],
    copts = [],
    defines = [],
    includes = [
        "src",
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [],
)
