# Description:
#   Kafka C/C++ (librdkafka) client library

licenses(["notice"])  # 2-clause BSD license

exports_files(["LICENSE"])

cc_library(
    name = "kafka",
    srcs = glob(
        [
            "src-cpp/*.h",
            "src-cpp/*.cpp",
            "src/*.c",
            "src/*.h",
        ],
        exclude = [
            "src/rddl.c",
            "src/rddl.h",
            "src/rdkafka_plugin.c",
            "src/rdkafka_plugin.h",
            "src/rdkafka_sasl_cyrus.c",
            "src/rdkafka_sasl_win32.c",
            "src/win32_config.h",
            "src/xxhash.c",
        ],
    ) + [
        "config/config.h",
        "config/src/set1_host.c",
        "config/src/win32_config.h",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "src/rdkafka_sasl_win32.c",
        ],
        "//conditions:default": [
        ],
    }),
    hdrs = [
        "config/config.h",
        "config/src/set1_host.c",
        "config/src/win32_config.h",
        "src/lz4.c",
        "src/xxhash.c",
    ],
    defines = [
        "LIBRDKAFKA_STATICLIB",
        "WIN32_LEAN_AND_MEAN",
        "XXH_PRIVATE_API",
    ],
    includes = [
        "config/src",
        "src",
        "src-cpp",
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "@boringssl//:ssl",
        "@zlib",
        "@zstd",
    ],
)

genrule(
    name = "set1_host_c",
    outs = ["config/src/set1_host.c"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#include <openssl/ssl.h>",
        "int SSL_set1_host(SSL *s, const char *hostname) {",
        "  return X509_VERIFY_PARAM_set1_host(SSL_get0_param(s), hostname, 0);",
        "}",
        "EOF",
    ]),
)

genrule(
    name = "win32_config_h",
    outs = ["config/src/win32_config.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define WITH_SSL 1",
        "#define WITH_ZLIB 1",
        "#define WITH_SNAPPY 1",
        "#define WITH_ZSTD 1",
        "#define WITH_ZSTD_STATIC 1",
        "#define WITH_LZ4_EXT 1",
        "#define WITH_SASL 1",
        "#define WITH_SASL_SCRAM 1",
        "#define WITH_SASL_OAUTHBEARER 1",
        "#define WITH_HDRHISTOGRAM 1",
        "#define WITH_PLUGINS 0",
        "#define ENABLE_DEVEL 0",
        "#define BUILT_WITH \"SSL ZLIB SNAPPY ZSTD LZ4 SASL SASL_SCRAM SASL_OAUTHBEARER HDRHISTOGRAM\"",
        "EOF",
    ]),
)

genrule(
    name = "config_h",
    outs = ["config/config.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define WITH_SSL 1",
        "#define WITH_ZLIB 1",
        "#define WITH_SNAPPY 1",
        "#define WITH_ZSTD 1",
        "#define WITH_ZSTD_STATIC 1",
        "#define WITH_LZ4_EXT 1",
        "#define WITH_SASL 1",
        "#define WITH_SASL_SCRAM 1",
        "#define WITH_SASL_OAUTHBEARER 1",
        "#define WITH_HDRHISTOGRAM 1",
        "#define WITH_PLUGINS 0",
        "#define ENABLE_DEVEL 0",
        "#define BUILT_WITH \"SSL ZLIB SNAPPY ZSTD LZ4 SASL SASL_SCRAM SASL_OAUTHBEARER HDRHISTOGRAM\"",
        "EOF",
    ]),
)
