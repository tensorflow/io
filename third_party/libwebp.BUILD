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
        "src/webp/*.h",
    ]) + [
        "imageio/imageio_util.c",
        "imageio/webpdec.c",
        "imageio/metadata.c",
        "imageio/webpdec.h",
        "imageio/metadata.h",
        "imageio/imageio_util.h",
        "examples/unicode.h",
    ],
    defines = [
        "WEBP_EXTERN=",
    ],
    includes = [
        "src",
    ],
    visibility = ["//visibility:public"],
)
