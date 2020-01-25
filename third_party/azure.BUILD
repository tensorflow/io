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
        "azure_storage_lite_EXPORTS",
        "USE_OPENSSL",
        "WIN32_LEAN_AND_MEAN",
        "_DEFAULT_SOURCE",
    ],
    includes = ["include"],
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [
            "-DEFAULTLIB:Rpcrt4.lib",
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
