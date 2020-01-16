# Description:
#   LMDB is the Lightning Memory-mapped Database.

licenses(["notice"])  # OpenLDAP Public License

exports_files(["LICENSE"])

cc_library(
    name = "lmdb",
    srcs = [
        "mdb.c",
        "midl.c",
    ],
    hdrs = [
        "lmdb.h",
        "midl.h",
    ],
    copts = [
        "-w",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [
            "-DEFAULTLIB:advapi32.lib",
        ],
        "//conditions:default": [
            "-lpthread",
        ],
    }),
    visibility = ["//visibility:public"],
)
