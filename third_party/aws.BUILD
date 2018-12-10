# Description:
#   AWS C++ SDK

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "aws",
    srcs = glob([
        "aws-cpp-sdk-core/source/platform/linux-shared/*.cpp",
        "aws-cpp-sdk-core/include/**/*.h",
        "aws-cpp-sdk-core/source/*.cpp",
        "aws-cpp-sdk-core/source/auth/**/*.cpp",
        "aws-cpp-sdk-core/source/config/**/*.cpp",
        "aws-cpp-sdk-core/source/client/**/*.cpp",
        "aws-cpp-sdk-core/source/external/**/*.cpp",
        "aws-cpp-sdk-core/source/internal/**/*.cpp",
        "aws-cpp-sdk-core/source/http/*.cpp",
        "aws-cpp-sdk-core/source/http/curl/**/*.cpp",
        "aws-cpp-sdk-core/source/http/standard/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/*.cpp",
        "aws-cpp-sdk-core/source/utils/base64/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/json/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/logging/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/memory/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/stream/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/threading/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/xml/**/*.cpp",
        "aws-cpp-sdk-core/source/utils/crypto/*.cpp",
        "aws-cpp-sdk-core/source/utils/crypto/factory/**/*.cpp",
        "aws-cpp-sdk-kinesis/include/**/*.h",
        "aws-cpp-sdk-kinesis/source/**/*.cpp",
        "aws-cpp-sdk-s3/include/**/*.h",
        "aws-cpp-sdk-s3/source/**/*.cpp",
    ]),
    hdrs = [
        "aws-cpp-sdk-core/include/aws/core/SDKConfig.h",
    ],
    defines = [
        "PLATFORM_LINUX",
        "ENABLE_CURL_CLIENT",
        "ENABLE_NO_ENCRYPTION",
    ],
    includes = [
        "aws-cpp-sdk-core/include/",
        "aws-cpp-sdk-kinesis/include/",
        "aws-cpp-sdk-s3/include/",
    ],
    deps = [
        "@curl",
    ],
)

genrule(
    name = "SDKConfig_h",
    srcs = [
        "aws-cpp-sdk-core/include/aws/core/SDKConfig.h.in",
    ],
    cmd = "sed 's/cmakedefine/define/g' $< > $@",
    outs = [
        "aws-cpp-sdk-core/include/aws/core/SDKConfig.h",
    ],
)
