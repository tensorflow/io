# Description:
#   HDF5 Library

licenses(["notice"])  # BSD-like license

exports_files(["COPYING"])

# H5Tinit.c/H5lib_settings.c/H5pubconf.h are generated separately with
# H5_HAVE_FILTER_SZIP/H5_HAVE_LIBSZ/H5_HAVE_SZLIB_H removed on Windows.
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
            "src/H5detect.c",
        ],
    ) + select({
        "@bazel_tools//src/conditions:windows": [
            "@org_tensorflow_io//third_party:hdf5/windows/H5Tinit.c",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "@org_tensorflow_io//third_party:hdf5/darwin/H5Tinit.c",
        ],
        "//conditions:default": [
            "@org_tensorflow_io//third_party:hdf5/linux/H5Tinit.c",
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
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow_io//third_party:hdf5",
        "@zlib",
    ],
)
