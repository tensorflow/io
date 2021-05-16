# Description:
#   azure-storage-fuse implements a FUSE filesystem for blob storage

licenses(["notice"])

cc_library(
    name = "azure",
    srcs = glob([
        "include/*.h",
        "include/blob/*.h",
        "include/todo/*.h",
        "include/http/*.h",
        "src/*.cpp",
        "src/http/*.cpp",
        "src/blob/*.cpp",
    ]),
    hdrs = [],
    defines = [
        "USE_OPENSSL",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "_WIN32",
            "WIN32",
            "WIN32_LEAN_AND_MEAN",
            'struct_stat="struct _stat64"',
        ],
        "//conditions:default": [
            "_DEFAULT_SOURCE",
            'struct_stat="struct stat"',
        ],
    }),
    includes = ["include"],
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [
            "-DEFAULTLIB:Rpcrt4.lib",
            "-DEFAULTLIB:Bcrypt.lib",
        ],
        "//conditions:default": [],
    }),
    textual_hdrs = [
        "include/constants.dat",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@curl",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin": [],
        "//conditions:default": [
            "@util_linux//:uuid",
            "@boringssl//:crypto",
        ],
    }),
)
