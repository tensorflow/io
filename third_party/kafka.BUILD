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
            "src/rdkafka_sasl_scram.c",
            "src/rdkafka_sasl_scram.h",
            "src/rdkafka_sasl_cyrus.c",
            "src/rdkafka_sasl_win32.c",
        ],
    ) + [
        "config/config.h",
        "config/set1_host/set1_host.c",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "src/rdkafka_sasl_win32.c",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "config/config.h",
        "config/set1_host/set1_host.c",
        "src/lz4.c",
    ],
    defines = [
        "LIBRDKAFKA_STATICLIB",
    ],
    includes = [
        "config/set1_host",
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
    name = "config_h",
    outs = ["config/config.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define WITH_SSL 1",
        "#define ENABLE_ZSTD 1",
        "#define ENABLE_SSL 1",
        "#define ENABLE_GSSAPI 1",
        "#define ENABLE_LZ4_EXT 1",
        "#define WITH_SASL_OAUTHBEARER 1",
        "#define BUILT_WITH \"BAZEL\"",
        "EOF",
    ]),
)

genrule(
    name = "set1_host_c",
    outs = ["config/set1_host/set1_host.c"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#include <openssl/ssl.h>",
        "int SSL_set1_host(SSL *s, const char *hostname) {",
        "  X509_VERIFY_PARAM *param;",
        "  param = SSL_get0_param(s);",
        "  return X509_VERIFY_PARAM_set1_host(param, hostname, 0);",
        "}",
        "EOF",
    ]),
)
