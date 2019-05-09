# Description:
#   HDF5 Library

licenses(["notice"])  # BSD-like license

exports_files(["COPYING"])

cc_library(
    name = "hdf5",
    srcs = glob(
        [
            "src/*.c",
            "hl/src/*.c",
            "c++/src/*.cpp",
        ],
        exclude = [
            "src/H5make_libsettings.c",
        ],
    ) + select({
        "@bazel_tools//src/conditions:darwin": [
            "darwin/src/H5pubconf.h",
            "darwin/src/H5lib_settings.c",
            "darwin/src/H5Tinit.c",
        ],
        "//conditions:default": [
            "linux/src/H5pubconf.h",
            "linux/src/H5lib_settings.c",
            "linux/src/H5Tinit.c",
        ],
    }),
    hdrs = glob([
        "src/*.h",
        "hl/src/*.h",
        "c++/src/*.h",
    ]),
    copts = [],
    includes = [
        "c++/src",
        "hl/src",
        "src",
    ] + select({
        "@bazel_tools//src/conditions:darwin": [
            "darwin/src",
        ],
        "//conditions:default": [
            "linux/src",
        ],
    }),
    linkopts = [],
    visibility = ["//visibility:public"],
)
