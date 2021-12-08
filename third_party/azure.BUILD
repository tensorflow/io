# Description:
#   azure-storage-fuse implements a FUSE filesystem for blob storage

licenses(["notice"])

cc_library(
    name = "azure",
    srcs = glob([
        "sdk/core/azure-core/inc/azure/*.hpp",
        "sdk/core/azure-core/inc/azure/core/*.hpp",
        "sdk/core/azure-core/inc/azure/core/credentials/*.hpp",
        "sdk/core/azure-core/inc/azure/core/cryptography/*.hpp",
        "sdk/core/azure-core/inc/azure/core/diagnostics/*.hpp",
        "sdk/core/azure-core/inc/azure/core/http/*.hpp",
        "sdk/core/azure-core/inc/azure/core/http/policies/*.hpp",
        "sdk/core/azure-core/inc/azure/core/internal/*.hpp",
        "sdk/core/azure-core/inc/azure/core/internal/cryptography/*.hpp",
        "sdk/core/azure-core/inc/azure/core/internal/diagnostics/*.hpp",
        "sdk/core/azure-core/inc/azure/core/internal/http/*.hpp",
        "sdk/core/azure-core/inc/azure/core/internal/io/*.hpp",
        "sdk/core/azure-core/inc/azure/core/internal/json/*.hpp",
        "sdk/core/azure-core/inc/azure/core/io/*.hpp",
        "sdk/core/azure-core/src/*.cpp",
        "sdk/core/azure-core/src/cryptography/*.cpp",
        "sdk/core/azure-core/src/http/*.cpp",
        "sdk/core/azure-core/src/io/*.cpp",
        "sdk/core/azure-core/src/private/*.hpp",
        "sdk/storage/azure-storage-blobs/inc/azure/storage/*.hpp",
        "sdk/storage/azure-storage-blobs/inc/azure/storage/blobs/*.hpp",
        "sdk/storage/azure-storage-blobs/inc/azure/storage/blobs/protocol/*.hpp",
        "sdk/storage/azure-storage-blobs/src/*.cpp",
        "sdk/storage/azure-storage-blobs/src/private/*.hpp",
        "sdk/storage/azure-storage-common/inc/azure/storage/common/*.hpp",
        "sdk/storage/azure-storage-common/inc/azure/storage/common/internal/*.hpp",
        "sdk/storage/azure-storage-common/src/*.cpp",
        "sdk/storage/azure-storage-common/src/private/*.hpp",
    ]) + select({
        "@bazel_tools//src/conditions:windows": [
            "sdk/core/azure-core/src/http/winhttp/win_http_transport.cpp",
        ],
        "//conditions:default": glob([
            "sdk/core/azure-core/src/http/curl/*.cpp",
            "sdk/core/azure-core/src/http/curl/*.hpp",
        ]),
    }),
    hdrs = [],
    defines = [] + select({
        "@bazel_tools//src/conditions:windows": [
            "BUILD_TRANSPORT_WINHTTP_ADAPTER",
        ],
        "//conditions:default": [
            "BUILD_CURL_HTTP_TRANSPORT_ADAPTER",
        ],
    }),
    includes = [
        "sdk/core/azure-core/inc/",
        "sdk/storage/azure-storage-blobs/inc/",
        "sdk/storage/azure-storage-common/inc/",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [
            "-DEFAULTLIB:Bcrypt.lib",
            "-DEFAULTLIB:Crypt32.lib",
            "-DEFAULTLIB:WebServices.lib",
            "-DEFAULTLIB:Winhttp.lib",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@boringssl//:crypto",
        "@boringssl//:ssl",
        "@libxml_archive//:libxml",
        "@zlib",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "@curl",
        ],
    }),
)
