# Description:
#   AWS C++ SDK

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "aws-sdk-cpp",
    srcs = glob([
        "aws-cpp-sdk-core/include/**/*.h",
        "aws-cpp-sdk-core/source/*.cpp",
        "aws-cpp-sdk-core/source/auth/**/*.cpp",
        "aws-cpp-sdk-core/source/client/**/*.cpp",
        "aws-cpp-sdk-core/source/config/**/*.cpp",
        "aws-cpp-sdk-core/source/external/**/*.cpp",
        "aws-cpp-sdk-core/source/http/*.cpp",
        "aws-cpp-sdk-core/source/http/curl/*.cpp",
        "aws-cpp-sdk-core/source/http/standard/*.cpp",
        "aws-cpp-sdk-core/source/internal/**/*.cpp",
        "aws-cpp-sdk-core/source/monitoring/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/*.cpp",
        "aws-cpp-sdk-core/source/utils/base64/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/crypto/*.cpp",
        "aws-cpp-sdk-core/source/utils/crypto/factory/*.cpp",
        "aws-cpp-sdk-core/source/utils/crypto/openssl/CryptoImpl.cpp",
        "aws-cpp-sdk-core/source/utils/event/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/json/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/logging/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/memory/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/stream/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/threading/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/xml/**/*.cpp",
        "aws-cpp-sdk-kinesis/include/**/*.h",
        "aws-cpp-sdk-kinesis/source/**/*.cpp",
        "aws-cpp-sdk-s3/include/**/*.h",
        "aws-cpp-sdk-s3/source/**/*.cpp",
        "aws-cpp-sdk-transfer/include/**/*.h",
        "aws-cpp-sdk-transfer/source/**/*.cpp",
    ]) + select({
        "@bazel_tools//src/conditions:windows": glob([
            "aws-cpp-sdk-core/source/http/windows/*.cpp",
            "aws-cpp-sdk-core/source/net/windows/*.cpp",
            "aws-cpp-sdk-core/source/platform/windows/*.cpp",
        ]),
        "//conditions:default": glob([
            "aws-cpp-sdk-core/source/net/linux-shared/*.cpp",
            "aws-cpp-sdk-core/source/platform/linux-shared/*.cpp",
        ]),
    }),
    hdrs = [
        "aws-cpp-sdk-core/include/aws/core/SDKConfig.h",
    ],
    defines = [
        'AWS_SDK_VERSION_STRING=\\"1.8.105\\"',
        "AWS_SDK_VERSION_MAJOR=1",
        "AWS_SDK_VERSION_MINOR=8",
        "AWS_SDK_VERSION_PATCH=105",
        "ENABLE_OPENSSL_ENCRYPTION=1",
        "ENABLE_CURL_CLIENT=1",
        "OPENSSL_IS_BORINGSSL=1",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "PLATFORM_WINDOWS",
            "WIN32_LEAN_AND_MEAN",
        ],
        "//conditions:default": [
            "PLATFORM_LINUX",
        ],
    }),
    includes = [
        "aws-cpp-sdk-core/include",
        "aws-cpp-sdk-kinesis/include",
        "aws-cpp-sdk-s3/include",
        "aws-cpp-sdk-transfer/include",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "aws-cpp-sdk-core/include/aws/core/platform/refs",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [
            "-DEFAULTLIB:userenv.lib",
            "-DEFAULTLIB:version.lib",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@aws-c-common",
        "@aws-c-event-stream",
        "@aws-checksums",
        "@boringssl//:crypto",
        "@curl",
    ],
)

genrule(
    name = "SDKConfig_h",
    outs = [
        "aws-cpp-sdk-core/include/aws/core/SDKConfig.h",
    ],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define USE_AWS_MEMORY_MANAGEMENT",
        "#if defined(_MSC_VER)",
        "#include <Windows.h>",
        "#undef IGNORE",
        "#endif",
        "EOF",
    ]),
)
