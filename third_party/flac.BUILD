package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD license for libFLAC

cc_library(
    name = "flac",
    srcs = glob(
        [
            "src/libFLAC/*.c",
            "src/libFLAC/include/private/*.h",
            "src/libFLAC/include/protected/*.h",
        ],
        exclude = [
            "src/libFLAC/windows_unicode_filenames.c",
        ],
    ) + [
        "config/config.h",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "src/libFLAC/windows_unicode_filenames.c",
        ],
        "//conditions:default": [],
    }),
    hdrs = glob(
        [
            "include/FLAC/*.h",
            "include/share/*.h",
        ],
    ),
    copts = [],
    defines = [
        "HAVE_CONFIG_H",
        "FLAC__NO_DLL",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "HAVE_BSWAP16=1" +
            "HAVE_BSWAP32=1",
        ],
    }),
    includes = [
        "config",
        "include",
        "src/libFLAC/include",
    ],
    deps = [
        "@ogg",
    ],
)

genrule(
    name = "config_h",
    srcs = ["config.cmake.h.in"],
    outs = ["config/config.h"],
    cmd = select({
        "@platforms//cpu:x86_64": (
            "sed " +
            "-e 's/cmakedefine01 CPU_IS_BIG_ENDIAN/define CPU_IS_BIG_ENDIAN 0/g' " +
            "-e 's/cmakedefine01 CPU_IS_LITTLE_ENDIAN/define CPU_IS_LITTLE_ENDIAN 0/g' " +
            "-e 's/cmakedefine01 ENABLE_64_BIT_WORDS/define ENABLE_64_BIT_WORDS 0/g' " +
            "-e 's/cmakedefine01 OGG_FOUND/define OGG_FOUND 1/g' " +
            "-e 's/cmakedefine01 FLAC__HAS_X86INTRIN/define FLAC__HAS_X86INTRIN 1/g' " +
            "-e 's/cmakedefine01 WITH_AVX/define WITH_AVX 1/g' " +
            "-e 's/cmakedefine HAVE_CLOCK_GETTIME/define HAVE_CLOCK_GETTIME/g' " +
            "-e 's/cmakedefine HAVE_CPUID_H/define HAVE_CPUID_H/g' " +
            "-e 's/cmakedefine HAVE_CXX_VARARRAYS/define HAVE_CXX_VARARRAYS/g' " +
            "-e 's/cmakedefine HAVE_FSEEKO/define HAVE_FSEEKO/g' " +
            "-e 's/cmakedefine01 HAVE_INTTYPES_H/define HAVE_INTTYPES_H 1/g' " +
            "-e 's/cmakedefine01 HAVE_LROUND/define HAVE_LROUND 1/g' " +
            "-e 's/cmakedefine01 HAVE_STDINT_H/define HAVE_STDINT_H 1/g' " +
            "-e 's/cmakedefine HAVE_STRING_H/define HAVE_STRING_H/g' " +
            "-e 's/cmakedefine HAVE_SYS_IOCTL_H/define HAVE_SYS_IOCTL_H/g' " +
            "-e 's/cmakedefine HAVE_SYS_TYPES_H/define HAVE_SYS_TYPES_H/g' " +
            "-e 's/cmakedefine HAVE_TERMIOS_H/define HAVE_TERMIOS_H/g' " +
            "-e 's/cmakedefine NDEBUG/define NDEBUG/g' " +
            "-e 's/@PROJECT_VERSION@/1.3.3/g' " +
            "-e 's/cmakedefine DODEFINE_XOPEN_SOURCE 500/undef DODEFINE_XOPEN_SOURCE/g' " +
            "-e 's/cmakedefine DODEFINE_FORTIFY_SOURCE/define DODEFINE_FORTIFY_SOURCE/g' " +
            "-e 's/cmakedefine DODEFINE_EXTENSIONS/define DODEFINE_EXTENSIONS/g' " +
            "-e 's/cmakedefine01/undef/g' " +
            "-e 's/cmakedefine/undef/g' " +
            "$< >$@"
        ),
        "//conditions:default": (
            "sed " +
            "-e 's/cmakedefine01 CPU_IS_BIG_ENDIAN/define CPU_IS_BIG_ENDIAN 0/g' " +
            "-e 's/cmakedefine01 CPU_IS_LITTLE_ENDIAN/define CPU_IS_LITTLE_ENDIAN 0/g' " +
            "-e 's/cmakedefine01 ENABLE_64_BIT_WORDS/define ENABLE_64_BIT_WORDS 0/g' " +
            "-e 's/cmakedefine01 OGG_FOUND/define OGG_FOUND 1/g' " +
            "-e 's/cmakedefine HAVE_CLOCK_GETTIME/define HAVE_CLOCK_GETTIME/g' " +
            "-e 's/cmakedefine HAVE_CXX_VARARRAYS/define HAVE_CXX_VARARRAYS/g' " +
            "-e 's/cmakedefine HAVE_FSEEKO/define HAVE_FSEEKO/g' " +
            "-e 's/cmakedefine01 HAVE_INTTYPES_H/define HAVE_INTTYPES_H 1/g' " +
            "-e 's/cmakedefine01 HAVE_LROUND/define HAVE_LROUND 1/g' " +
            "-e 's/cmakedefine01 HAVE_STDINT_H/define HAVE_STDINT_H 1/g' " +
            "-e 's/cmakedefine HAVE_STRING_H/define HAVE_STRING_H/g' " +
            "-e 's/cmakedefine HAVE_SYS_IOCTL_H/define HAVE_SYS_IOCTL_H/g' " +
            "-e 's/cmakedefine HAVE_SYS_TYPES_H/define HAVE_SYS_TYPES_H/g' " +
            "-e 's/cmakedefine HAVE_TERMIOS_H/define HAVE_TERMIOS_H/g' " +
            "-e 's/cmakedefine NDEBUG/define NDEBUG/g' " +
            "-e 's/@PROJECT_VERSION@/1.3.3/g' " +
            "-e 's/cmakedefine DODEFINE_XOPEN_SOURCE 500/undef DODEFINE_XOPEN_SOURCE/g' " +
            "-e 's/cmakedefine DODEFINE_FORTIFY_SOURCE/define DODEFINE_FORTIFY_SOURCE/g' " +
            "-e 's/cmakedefine DODEFINE_EXTENSIONS/define DODEFINE_EXTENSIONS/g' " +
            "-e 's/cmakedefine01/undef/g' " +
            "-e 's/cmakedefine/undef/g' " +
            "$< >$@"
        ),
    }),
)
