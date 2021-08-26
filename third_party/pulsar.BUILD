# Description:
#   Pulsar C++ client library

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

proto_library(
    name = "PulsarApi_proto",
    srcs = ["pulsar-client-cpp/lib/PulsarApi.proto"],
)

cc_proto_library(
    name = "PulsarApi_cc_proto",
    deps = [":PulsarApi_proto"],
)

cc_library(
    name = "pulsar",
    srcs = glob([
        "pulsar-client-cpp/include/pulsar/*.h",
        "pulsar-client-cpp/lib/*.h",
        "pulsar-client-cpp/lib/*.cc",
        "pulsar-client-cpp/lib/auth/**/*.h",
        "pulsar-client-cpp/lib/auth/**/*.cc",
        # lz4 is imported in another package, so we just add the header here
        "pulsar-client-cpp/lib/lz4/*.h",
        "pulsar-client-cpp/lib/checksum/*.h",
        "pulsar-client-cpp/lib/checksum/*.hpp",
        "pulsar-client-cpp/lib/checksum/*.cc",
        "pulsar-client-cpp/lib/stats/*.h",
        "pulsar-client-cpp/lib/stats/*.cc",
    ]),
    copts = select({
        "@platforms//cpu:x86_64": [
            "-msse4.2",
            "-mpclmul",
        ],
        "//conditions:default": [],
    }),
    defines = [
        "_PULSAR_VERSION_=\\\"2.6.1\\\"",
        "WIN32_LEAN_AND_MEAN",
        "PULSAR_STATIC",
    ],
    includes = [
        "pulsar-client-cpp",
        "pulsar-client-cpp/include",
        "pulsar-client-cpp/lib",
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
