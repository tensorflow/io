# Description:
#   A library for decoding and encoding GIF images

licenses(["notice"])  # MIT

exports_files(["COPYING"])

cc_library(
    name = "giflib",
    srcs = [
        "dgif_lib.c",
        "egif_lib.c",
        "gif_err.c",
        "gif_font.c",
        "gif_hash.c",
        "gif_hash.h",
        "gif_lib.h",
        "gif_lib_private.h",
        "gifalloc.c",
        "openbsd-reallocarray.c",
        "quantize.c",
    ],
    hdrs = select({
        "@bazel_tools//src/conditions:windows": [
            "windows/unistd.h",
        ],
        "//conditions:default": [],
    }),
    copts = [],
    includes = ["windows"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "windows_unistd_h",
    outs = ["windows/unistd.h"],
    cmd = "touch $@",
)
