package(default_visibility = ["//visibility:public"])

cc_library(
    name = "openexr",
    srcs = [
        "openexr/b44ExpLogTable.h",
        "openexr/dwaLookups.h",
        "openexr/eLut.h",
        "openexr/toFloat.h",
    ],
    hdrs = [],
    copts = [],
    includes = ["openexr"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "font",
    srcs = [
        "font/font_opensans_regular.h",
    ],
    hdrs = [],
    copts = [],
    includes = ["font"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "hdf5",
    srcs = select({
        "@bazel_tools//src/conditions:windows": [
            "hdf5/windows/H5lib_settings.c",
            "hdf5/windows/H5pubconf.h",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "hdf5/darwin/H5lib_settings.c",
            "hdf5/darwin/H5pubconf.h",
        ],
        "//conditions:default": [
            "hdf5/linux/H5lib_settings.c",
            "hdf5/linux/H5pubconf.h",
        ],
    }),
    hdrs = [],
    copts = [],
    includes = select({
        "@bazel_tools//src/conditions:windows": [
            "hdf5/windows",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "hdf5/darwin",
        ],
        "//conditions:default": [
            "hdf5/linux",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [],
)

exports_files([
    "hdf5/windows/H5Tinit.c",
    "hdf5/darwin/H5Tinit.c",
    "hdf5/linux/H5Tinit.c",
])
