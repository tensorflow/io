# Description:
#   uuid

licenses(["notice"])

cc_library(
    name = "uuid",
    srcs = [
        "include/all-io.h",
        "include/c.h",
        "include/md5.h",
        "include/nls.h",
        "include/randutils.h",
        "include/sha1.h",
        "include/strutils.h",
        "lib/md5.c",
        "lib/randutils.c",
        "lib/sha1.c",
        "libuuid/src/clear.c",
        "libuuid/src/compare.c",
        "libuuid/src/copy.c",
        "libuuid/src/gen_uuid.c",
        "libuuid/src/isnull.c",
        "libuuid/src/pack.c",
        "libuuid/src/parse.c",
        "libuuid/src/predefined.c",
        "libuuid/src/unpack.c",
        "libuuid/src/unparse.c",
        "libuuid/src/uuid.h",
        "libuuid/src/uuidP.h",
        "libuuid/src/uuid_time.c",
        "libuuid/src/uuidd.h",
    ],
    hdrs = [
        "libuuid/src/uuid.h",
    ],
    copts = ["-std=c99"],
    defines = select({
        "//conditions:default": [
            "_XOPEN_SOURCE=700",
            "HAVE_NANOSLEEP",
            "HAVE_SYS_FILE_H",
            "HAVE_MEMCPY",
            "HAVE_STRNLEN",
            "HAVE_STRNDUP",
            "HAVE_STRNCHR",
        ],
    }),
    include_prefix = "uuid",
    includes = ["include"],
    strip_include_prefix = "libuuid/src",
    visibility = ["//visibility:public"],
)
