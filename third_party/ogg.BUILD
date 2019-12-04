package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD license

cc_library(
    name = "ogg",
    srcs = glob([
        "src/*.h",
        "src/*.c",
    ]),
    hdrs = glob([
        "include/ogg/*.h",
    ]) + [
        "include/ogg/config_types.h",
    ],
    copts = [
    ],
    includes = [
        "include",
    ],
)

genrule(
    name = "config_types_h",
    srcs = ["include/ogg/config_types.h.in"],
    outs = ["include/ogg/config_types.h"],
    cmd = ("sed " +
           "-e 's/@INCLUDE_INTTYPES_H@/1/g' " +
           "-e 's/@INCLUDE_STDINT_H@/1/g' " +
           "-e 's/@INCLUDE_SYS_TYPES_H@/1/g' " +
           "-e 's/@SIZE16@/int16_t/g' " +
           "-e 's/@USIZE16@/uint16_t/g' " +
           "-e 's/@SIZE32@/int32_t/g' " +
           "-e 's/@USIZE32@/uint32_t/g' " +
           "-e 's/@SIZE64@/int64_t/g' " +
           "-e 's/@USIZE64@/uint64_t/g' " +
           "$< >$@"),
)
