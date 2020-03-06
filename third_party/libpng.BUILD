# Description:
#   libpng library

licenses(["notice"])  # libpng license

exports_files(["LICENSE"])

cc_library(
    name = "libpng",
    srcs = glob([
        "*.c",
        "*.h",
    ]),
    hdrs = [
        "config/pnglibconf.h",
    ],
    includes = [
        ".",
        "config",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@zlib",
    ],
)

genrule(
    name = "pnglibconf_h",
    srcs = ["scripts/pnglibconf.h.prebuilt"],
    outs = ["config/pnglibconf.h"],
    cmd = ("sed " +
           "-e 's/define PNG_ZLIB_VERNUM 0/define PNG_ZLIB_VERNUM 0x12b0/g' " +
           "$< >$@"),
)
