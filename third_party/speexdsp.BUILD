package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD license

cc_library(
    name = "speexdsp",
    srcs = glob(
        [
            "libspeexdsp/*.c",
            "libspeexdsp/*.h",
            "include/speex/*.h",
        ],
        exclude = [
            "libspeexdsp/test*.c",
        ],
    ),
    hdrs = select({
        "@bazel_tools//src/conditions:windows": [
            "win32/config.h",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "config/config.h",
        ],
        "//conditions:default": [
            "config/config.h",
            "config/speexdsp_config_types.h",
        ],
    }),
    copts = [],
    defines = [
        "HAVE_CONFIG_H",
    ],
    includes = [
        "include",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "win32",
        ],
        "//conditions:default": [
            "config",
        ],
    }),
    deps = [],
)

genrule(
    name = "config_h",
    outs = ["config/config.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define EXPORT",
        "#define FLOATING_POINT",
        "#define USE_SMALLFT",
        "#if defined __x86_64__",
        "#define USE_SSE",
        "#define USE_SSE2",
        "#endif",
        "#define VAR_ARRAYS",
        "#define restrict __restrict",
        "EOF",
    ]),
)

genrule(
    name = "speexdsp_config_types_h",
    srcs = ["include/speex/speexdsp_config_types.h.in"],
    outs = ["config/speexdsp_config_types.h"],
    cmd = ("sed " +
           "-e 's/@INCLUDE_STDINT@/#include <stdint.h>/g' " +
           "-e 's/@SIZE16@/int16_t/g' " +
           "-e 's/@USIZE16@/uint16_t/g' " +
           "-e 's/@SIZE32@/int32_t/g' " +
           "-e 's/@USIZE32@/uint32_t/g' " +
           "$< >$@"),
)
