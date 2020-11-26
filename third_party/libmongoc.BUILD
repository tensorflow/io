# Description:
#   Mongo C++ client library

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "libmongoc",
    srcs = glob([
        "src/libmongoc/src/mongoc/**/*.c",
        "src/libbson/src/bson/**/*.c",
        "src/libbson/src/jsonsl/**/*.c",
        "src/common/*.c",
        "src/kms-message/src/**/*.c",
    ]),
    hdrs = [
        "src/libmongoc/src/mongoc/mongoc-config.h",
        "src/libbson/src/bson/bson-config.h",
    ] + glob(
        [
            "src/libmongoc/src/mongoc/**/*.h",
            "src/libmongoc/src/mongoc/**/*.hh",
            "src/libmongoc/src/mongoc/**/*.def",
            "src/libmongoc/src/mongoc/**/*.defs",
            "src/libbson/src/bson/**/*.h",
            "src/libbson/src/jsonsl/**/*.h",
            "src/common/*.h",
            "src/kms-message/src/**/*.h",
        ],
    ) + select({
        "@bazel_tools//src/conditions:windows": [
            "windows/unistd.h",
        ],
        "//conditions:default": [],
    }),
    copts = [],
    defines = [
        "MONGOC_COMPILATION",
        "BSON_COMPILATION",
    ],
    includes = [
        "src",
        "src/common",
        "src/kms-message",
        "src/kms-message/src",
        "src/libbson/src",
        "src/libbson/src/bson",
        "src/libbson/src/jsonsl",
        "src/libmongoc/src",
        "src/libmongoc/src/mongoc",
        "windows",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],
        "@bazel_tools//src/conditions:windows": [
            # https://jira.mongodb.org/browse/CXX-1731
            "-DEFAULTLIB:ws2_32.lib",
            "-DEFAULTLIB:advapi32.lib",
            "-DEFAULTLIB:crypt32.lib",
            "-DEFAULTLIB:Normaliz.lib",
        ],
        "//conditions:default": [
            "-lrt",
            "-lresolv",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@boost",
        "@boringssl//:crypto",
        "@boringssl//:ssl",
        "@snappy",
        "@zlib",
        "@zstd",
    ],
)

base_config = (
    "-e 's/@MONGOC_ENABLE_SSL_OPENSSL@/0/g' " +
    "-e 's/@MONGOC_ENABLE_SSL_LIBRESSL@/0/g' " +
    "-e 's/@MONGOC_ENABLE_SSL_SECURE_CHANNEL@/0/g' " +
    "-e 's/@MONGOC_ENABLE_CRYPTO_CNG@/0/g' " +
    "-e 's/@MONGOC_ENABLE_CRYPTO_LIBCRYPTO@/0/g' " +
    "-e 's/@MONGOC_ENABLE_CRYPTO_SYSTEM_PROFILE@/0/g' " +
    "-e 's/@MONGOC_HAVE_ASN1_STRING_GET0_DATA@/0/g' " +
    "-e 's/@MONGOC_HAVE_SASL_CLIENT_DONE@/0/g' " +
    "-e 's/@MONGOC_NO_AUTOMATIC_GLOBALS@/1/g' " +
    "-e 's/@MONGOC_HAVE_SOCKLEN@/1/g' " +
    "-e 's/@MONGOC_ENABLE_COMPRESSION@/1/g' " +
    "-e 's/@MONGOC_ENABLE_COMPRESSION_SNAPPY@/1/g' " +
    "-e 's/@MONGOC_ENABLE_COMPRESSION_ZLIB@/1/g' " +
    "-e 's/@MONGOC_ENABLE_COMPRESSION_ZSTD@/1/g' " +
    "-e 's/@MONGOC_ENABLE_SHM_COUNTERS@/0/g' " +
    "-e 's/@MONGOC_ENABLE_RDTSCP@/0/g' " +
    "-e 's/@MONGOC_HAVE_SCHED_GETCPU@/0/g' " +
    "-e 's/@MONGOC_TRACE@/0/g' " +
    "-e 's/@MONGOC_ENABLE_ICU@/0/g' " +
    "-e 's/@MONGOC_ENABLE_CLIENT_SIDE_ENCRYPTION@/0/g' " +
    "-e 's/@MONGOC_HAVE_SS_FAMILY@/1/g' " +
    "-e 's/@MONGOC_HAVE_RES_NSEARCH@/1/g' " +
    "-e 's/@MONGOC_HAVE_RES_SEARCH@/0/g' " +
    "-e 's/@MONGOC_ENABLE_MONGODB_AWS_AUTH@/1/g' " +
    "-e 's/@MONGOC_SOCKET_ARG3@/socklen_t/g' " +
    "-e 's/@MONGOC_SOCKET_ARG2@/struct sockaddr/g' " +
    "$< >$@"
)

genrule(
    name = "mongoc_config_h",
    srcs = ["src/libmongoc/src/mongoc/mongoc-config.h.in"],
    outs = ["src/libmongoc/src/mongoc/mongoc-config.h"],
    cmd = ("sed " +
           "-e 's/@MONGOC_USER_SET_CFLAGS@//g' " +
           "-e 's/@MONGOC_USER_SET_LDFLAGS@//g' ") +
          select({
              "@bazel_tools//src/conditions:windows": (
                  "-e 's/@MONGOC_ENABLE_SSL@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SSL_SECURE_TRANSPORT@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_CRYPTO@/1/g' " +
                  "-e 's/@MONGOC_ENABLE_CRYPTO_COMMON_CRYPTO@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SASL@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SASL_CYRUS@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SASL_SSPI@/0/g' " +
                  "-e 's/@MONGOC_HAVE_DNSAPI@/1/g' " +
                  "-e 's/@MONGOC_HAVE_RES_NDESTROY@/0/g' " +
                  "-e 's/@MONGOC_HAVE_RES_NCLOSE@/0/g' "
              ),
              "@bazel_tools//src/conditions:darwin": (
                  "-e 's/@MONGOC_ENABLE_SSL@/1/g' " +
                  "-e 's/@MONGOC_ENABLE_SSL_SECURE_TRANSPORT@/1/g' " +
                  "-e 's/@MONGOC_ENABLE_CRYPTO@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_CRYPTO_COMMON_CRYPTO@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SASL@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SASL_CYRUS@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SASL_SSPI@/0/g' " +
                  "-e 's/@MONGOC_HAVE_DNSAPI@/0/g' " +
                  "-e 's/@MONGOC_HAVE_RES_NDESTROY@/1/g' " +
                  "-e 's/@MONGOC_HAVE_RES_NCLOSE@/0/g' "
              ),
              "//conditions:default": (
                  "-e 's/@MONGOC_ENABLE_SSL@/1/g' " +
                  "-e 's/@MONGOC_ENABLE_SSL_SECURE_TRANSPORT@/1/g' " +
                  "-e 's/@MONGOC_ENABLE_CRYPTO@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_CRYPTO_COMMON_CRYPTO@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SASL@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SASL_CYRUS@/0/g' " +
                  "-e 's/@MONGOC_ENABLE_SASL_SSPI@/0/g' " +
                  "-e 's/@MONGOC_HAVE_DNSAPI@/0/g' " +
                  "-e 's/@MONGOC_HAVE_RES_NDESTROY@/0/g' " +
                  "-e 's/@MONGOC_HAVE_RES_NCLOSE@/1/g' "
              ),
          }) +
          base_config,
)

genrule(
    name = "bson_config_h",
    srcs = ["src/libbson/src/bson/bson-config.h.in"],
    outs = ["src/libbson/src/bson/bson-config.h"],
    cmd = (
        "sed " +
        "-e 's/@BSON_HAVE_STRINGS_H@/1/g' " +
        "-e 's/@BSON_HAVE_STDBOOL_H@/1/g' " +
        "-e 's/@BSON_HAVE_ATOMIC_32_ADD_AND_FETCH@/1/g' " +
        "-e 's/@BSON_HAVE_ATOMIC_64_ADD_AND_FETCH@/1/g' " +
        "-e 's/@BSON_HAVE_CLOCK_GETTIME@/1/g' " +
        "-e 's/@BSON_HAVE_STRNLEN@/1/g' " +
        "-e 's/@BSON_HAVE_SNPRINTF@/1/g' " +
        "-e 's/@BSON_HAVE_GMTIME_R@/1/g' " +
        "-e 's/@BSON_HAVE_REALLOCF@/0/g' " +
        "-e 's/@BSON_HAVE_TIMESPEC@/1/g' " +
        "-e 's/@BSON_EXTRA_ALIGN@/1/g' " +
        "-e 's/@BSON_HAVE_SYSCALL_TID@/0/g' " +
        "-e 's/@BSON_HAVE_RAND_R@/1/g' " +
        "-e 's/@BSON_HAVE_STRLCPY@/1/g' " +
        "-e 's/@BSON_BYTE_ORDER@/1234/g' "
    ) + select({
        "@bazel_tools//src/conditions:windows": (
            "-e 's/@BSON_OS@/2/g' " +
            "$< >$@"
        ),
        "//conditions:default": (
            "-e 's/@BSON_OS@/1/g' " +
            "$< >$@"
        ),
    }),
)

genrule(
    name = "windows_unistd_h",
    outs = ["windows/unistd.h"],
    cmd = "touch $@",
)
