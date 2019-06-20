# Description:
#   uuid 

licenses(["notice"])

cc_library(
  name = "uuid",
  srcs = [
      "include/all-io.h",
      "include/c.h",
      "include/randutils.h",
      "include/strutils.h",
      "include/md5.h",
      "include/sha1.h",
      "libuuid/src/compare.c",
      "libuuid/src/unpack.c",
      "libuuid/src/isnull.c",
      "libuuid/src/copy.c",
      "libuuid/src/unparse.c",
      "libuuid/src/pack.c",
      "libuuid/src/clear.c",
      "libuuid/src/predefined.c",
      "libuuid/src/uuid_time.c",
      "libuuid/src/gen_uuid.c",
      "libuuid/src/parse.c",
      "libuuid/src/uuid.h",
      "libuuid/src/uuidd.h",
      "libuuid/src/uuidP.h",
      "lib/randutils.c",
      "include/nls.h",
      "lib/md5.c",
      "lib/sha1.c"
  ],
  hdrs = [
      "libuuid/src/uuid.h"
  ],
  copts = ["-std=c99"],
  strip_include_prefix = "libuuid/src",
  include_prefix = "uuid",
  defines = select({
      "//conditions:default": [
          "_XOPEN_SOURCE=700",
          "HAVE_NANOSLEEP",
          "HAVE_SYS_FILE_H",
          "HAVE_MEMCPY",
          "HAVE_STRNLEN",
          "HAVE_STRNDUP",
          "HAVE_STRNCHR"
      ],
  }),
  includes = ["include"],
  visibility = ["//visibility:public"]
)
