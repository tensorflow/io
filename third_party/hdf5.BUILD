# Description:
#   HDF5 Library

licenses(["notice"])  # BSD-like license

exports_files(["COPYING"])

cc_library(
    name = "hdf5",
    srcs = glob(
        [
            "src/H5pubconf.h",
            "src/*.c",
            "hl/src/*.c",
            "c++/src/*.cpp",
        ],
        exclude = [
            "src/H5make_libsettings.c",
        ],
    ),
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
    ],
    linkopts = [
    ],
    visibility = ["//visibility:public"],
)
