# Description:
#   AVIF library

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

cc_library(
    name = "libavif",
    srcs = glob(
        [
            "include/avif/*.h",
            "src/*.c",
        ],
        exclude = [
            "src/codec_aom.c",
            "src/codec_libgav1.c",
            "src/codec_rav1e.c",
        ],
    ),
    hdrs = [],
    defines = [
        #"AVIF_CODEC_AOM=1",
        "AVIF_CODEC_DAV1D=1",
        #"AVIF_CODEC_LIBGAV1=1",
        #"AVIF_CODEC_RAV1E=1",
    ],
    includes = [
        "include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@dav1d",
    ],
)
