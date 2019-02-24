# Description:
#   A library for decoding and encoding GIF images

licenses(["notice"])  # MIT

exports_files(["COPYING"])

cc_library(
    name = "giflib",
    srcs = [
        "lib/dgif_lib.c",
        "lib/egif_lib.c",
        "lib/gif_err.c",
        "lib/gif_font.c",
        "lib/gif_hash.c",
        "lib/gif_hash.h",
        "lib/gif_lib_private.h",
        "lib/gifalloc.c",
        "lib/openbsd-reallocarray.c",
        "lib/quantize.c",
    ],
    hdrs = [
        "lib/gif_lib.h",
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    includes = ["lib"],
    visibility = ["//visibility:public"],
)
