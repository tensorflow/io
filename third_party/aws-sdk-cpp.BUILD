# Description:
#   AWS C++ SDK

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "core",
    srcs = glob([
        "aws-cpp-sdk-core/source/*.cpp",  # AWS_SOURCE
        "aws-cpp-sdk-core/source/external/tinyxml2/*.cpp",  # AWS_TINYXML2_SOURCE
        "aws-cpp-sdk-core/source/external/cjson/*.cpp",  # CJSON_SOURCE
        "aws-cpp-sdk-core/source/auth/*.cpp",  # AWS_AUTH_SOURCE
        "aws-cpp-sdk-core/source/client/*.cpp",  # AWS_CLIENT_SOURCE
        "aws-cpp-sdk-core/source/internal/*.cpp",  # AWS_INTERNAL_SOURCE
        "aws-cpp-sdk-core/source/aws/model/*.cpp",  # AWS_MODEL_SOURCE
        "aws-cpp-sdk-core/source/http/*.cpp",  # HTTP_SOURCE
        "aws-cpp-sdk-core/source/http/standard/*.cpp",  # HTTP_STANDARD_SOURCE
        "aws-cpp-sdk-core/source/config/*.cpp",  # CONFIG_SOURCE
        "aws-cpp-sdk-core/source/monitoring/*.cpp",  # MONITORING_SOURCE
        "aws-cpp-sdk-core/source/utils/*.cpp",  # UTILS_SOURCE
        "aws-cpp-sdk-core/source/utils/event/*.cpp",  # UTILS_EVENT_SOURCE
        "aws-cpp-sdk-core/source/utils/base64/*.cpp",  # UTILS_BASE64_SOURCE
        "aws-cpp-sdk-core/source/utils/crypto/*.cpp",  # UTILS_CRYPTO_SOURCE
        "aws-cpp-sdk-core/source/utils/json/*.cpp",  # UTILS_JSON_SOURCE
        "aws-cpp-sdk-core/source/utils/threading/*.cpp",  # UTILS_THREADING_SOURCE
        "aws-cpp-sdk-core/source/utils/xml/*.cpp",  # UTILS_XML_SOURCE
        "aws-cpp-sdk-core/source/utils/logging/*.cpp",  # UTILS_LOGGING_SOURCE
        "aws-cpp-sdk-core/source/utils/memory/*.cpp",  # UTILS_MEMORY_SOURCE
        "aws-cpp-sdk-core/source/utils/memory/stl/*.cpp",  # UTILS_MEMORY_STL_SOURCE
        "aws-cpp-sdk-core/source/utils/stream/*.cpp",  # UTILS_STREAM_SOURCE
        "aws-cpp-sdk-core/source/utils/crypto/factory/*.cpp",  # UTILS_CRYPTO_FACTORY_SOURCE
        "aws-cpp-sdk-core/source/http/curl/*.cpp",  # HTTP_CURL_CLIENT_SOURCE
        "aws-cpp-sdk-core/source/utils/crypto/openssl/*.cpp",  # UTILS_CRYPTO_OPENSSL_SOURCE
    ]) + select({
        "@bazel_tools//src/conditions:windows": glob([
            "aws-cpp-sdk-core/source/net/windows/*.cpp",  # NET_SOURCE
            "aws-cpp-sdk-core/source/platform/windows/*.cpp",  # PLATFORM_WINDOWS_SOURCE
        ]),
        "//conditions:default": glob([
            "aws-cpp-sdk-core/source/net/linux-shared/*.cpp",  # NET_SOURCE
            "aws-cpp-sdk-core/source/platform/linux-shared/*.cpp",  # PLATFORM_LINUX_SHARED_SOURCE
        ]),
    }),
    hdrs = [
        "aws-cpp-sdk-core/include/aws/core/SDKConfig.h",
    ] + glob([
        "aws-cpp-sdk-core/include/aws/core/*.h",  # AWS_HEADERS
        "aws-cpp-sdk-core/include/aws/core/auth/*.h",  # AWS_AUTH_HEADERS
        "aws-cpp-sdk-core/include/aws/core/client/*.h",  # AWS_CLIENT_HEADERS
        "aws-cpp-sdk-core/include/aws/core/internal/*.h",  # AWS_INTERNAL_HEADERS
        "aws-cpp-sdk-core/include/aws/core/net/*.h",  # NET_HEADERS
        "aws-cpp-sdk-core/include/aws/core/http/*.h",  # HTTP_HEADERS
        "aws-cpp-sdk-core/include/aws/core/http/standard/*.h",  # HTTP_STANDARD_HEADERS
        "aws-cpp-sdk-core/include/aws/core/config/*.h",  # CONFIG_HEADERS
        "aws-cpp-sdk-core/include/aws/core/monitoring/*.h",  # MONITORING_HEADERS
        "aws-cpp-sdk-core/include/aws/core/platform/*.h",  # PLATFORM_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/*.h",  # UTILS_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/event/*.h",  # UTILS_EVENT_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/base64/*.h",  # UTILS_BASE64_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/crypto/*.h",  # UTILS_CRYPTO_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/json/*.h",  # UTILS_JSON_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/threading/*.h",  # UTILS_THREADING_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/xml/*.h",  # UTILS_XML_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/memory/*.h",  # UTILS_MEMORY_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/memory/stl/*.h",  # UTILS_STL_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/logging/*.h",  # UTILS_LOGGING_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/ratelimiter/*.h",  # UTILS_RATE_LIMITER_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/stream/*.h",  # UTILS_STREAM_HEADERS
        "aws-cpp-sdk-core/include/aws/core/external/cjson/*.h",  # CJSON_HEADERS
        "aws-cpp-sdk-core/include/aws/core/external/tinyxml2/*.h",  # TINYXML2_HEADERS
        "aws-cpp-sdk-core/include/aws/core/http/curl/*.h",  # HTTP_CURL_CLIENT_HEADERS
        "aws-cpp-sdk-core/include/aws/core/utils/crypto/openssl/*.h",  # UTILS_CRYPTO_OPENSSL_HEADERS
    ]),
    defines = [
        'AWS_SDK_VERSION_STRING=\\"1.8.187\\"',
        "AWS_SDK_VERSION_MAJOR=1",
        "AWS_SDK_VERSION_MINOR=8",
        "AWS_SDK_VERSION_PATCH=187",
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
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [
            "-DEFAULTLIB:userenv.lib",
            "-DEFAULTLIB:version.lib",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@aws-c-event-stream",
        "@boringssl//:crypto",
        "@boringssl//:ssl",
        "@curl",
    ],
)

cc_library(
    name = "s3",
    srcs = glob([
        "aws-cpp-sdk-s3/source/*.cpp",  # AWS_S3_SOURCE
        "aws-cpp-sdk-s3/source/model/*.cpp",  # AWS_S3_MODEL_SOURCE
    ]),
    hdrs = glob([
        "aws-cpp-sdk-s3/include/aws/s3/*.h",  # AWS_S3_HEADERS
        "aws-cpp-sdk-s3/include/aws/s3/model/*.h",  # AWS_S3_MODEL_HEADERS
    ]),
    includes = [
        "aws-cpp-sdk-s3/include",
    ],
    deps = [
        ":core",
    ],
)

cc_library(
    name = "transfer",
    srcs = glob([
        "aws-cpp-sdk-transfer/source/transfer/*.cpp",  # TRANSFER_SOURCE
    ]),
    hdrs = glob([
        "aws-cpp-sdk-transfer/include/aws/transfer/*.h",  # TRANSFER_HEADERS
    ]),
    includes = [
        "aws-cpp-sdk-transfer/include",
    ],
    deps = [
        ":core",
        ":s3",
    ],
)

cc_library(
    name = "kinesis",
    srcs = glob([
        "aws-cpp-sdk-kinesis/source/*.cpp",  # AWS_KINESIS_SOURCE
        "aws-cpp-sdk-kinesis/source/model/*.cpp",  # AWS_KINESIS_MODEL_SOURCE
    ]),
    hdrs = glob([
        "aws-cpp-sdk-kinesis/include/aws/kinesis/*.h",  # AWS_KINESIS_HEADERS
        "aws-cpp-sdk-kinesis/include/aws/kinesis/model/*.h",  # AWS_KINESIS_MODEL_HEADERS
    ]),
    includes = [
        "aws-cpp-sdk-kinesis/include",
    ],
    deps = [
        ":core",
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
