# Description:
#   Pulsar C++ client library

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

proto_library(
    name = "PulsarApi_proto",
    srcs = ["proto/PulsarApi.proto"],
)

cc_proto_library(
    name = "PulsarApi_cc_proto",
    deps = [":PulsarApi_proto"],
)

cc_library(
    name = "pulsar",
    srcs = glob(
        [
            "include/pulsar/*.h",
            "lib/*.h",
            "lib/*.cc",
            "lib/auth/**/*.h",
            "lib/auth/**/*.cc",
            "lib/lz4/*.h",
            "lib/lz4/*.cc",
            "lib/checksum/*.h",
            "lib/checksum/*.hpp",
            "lib/checksum/*.cc",
            "lib/stats/*.h",
            "lib/stats/*.cc",
        ],
        exclude = [
            "lib/checksum/crc32c_sse42.cc",
        ],
    ) + select({
        "@platforms//cpu:x86_64": [
            "lib/checksum/crc32c_sse42.cc",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        # declare header from above genrule
        "include/pulsar/Version.h",
    ],
    copts = select({
        "@platforms//cpu:x86_64": [
            "-msse4.2",
            "-mpclmul",
        ],
        "//conditions:default": [],
    }),
    defines = [
        "WIN32_LEAN_AND_MEAN",
        "PULSAR_STATIC",
    ],
    includes = [
        "include",
        "lib",
        "proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":PulsarApi_cc_proto",
        "@boost",
        "@boringssl//:crypto",
        "@boringssl//:ssl",
        "@curl",
        "@zstd",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "@dlfcn-win32",
        ],
        "//conditions:default": [],
    }),
)

genrule(
    name = "Version_h",
    srcs = ["templates/Version.h.in"],
    outs = ["include/pulsar/Version.h"],
    cmd = ("sed " +
           "-e 's/@PULSAR_CLIENT_VERSION_MACRO@/3003000/g' " +
           "-e 's/@PULSAR_CLIENT_VERSION@/3.3.0/g' " +
           "$< >$@"),
)
