# Description:
#   dav1d library

licenses(["notice"])  # BSD license

exports_files(["COPYING"])

cc_library(
    name = "dav1d",
    visibility = ["//visibility:public"],
    deps = [
        ":dav1d16",
        ":dav1d8",
    ],
)

cc_library(
    name = "dav1d8",
    srcs = glob(
        [
            "include/dav1d/*.h",
            "include/common/*.h",
            "src/*.c",
            "src/x86/*.c",
            "src/x86/*.h",
            "src/*.h",
        ],
        exclude = [
            "src/x86/msac_init.c",
        ],
    ) + select({
        "@bazel_tools//src/conditions:windows": [
            "include/compat/msvc/stdatomic.h",
            "src/win32/thread.c",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "build/config.h",
        "build/vcs_version.h",
        "build/version.h",
    ],
    copts = [
        "-std=c99",
    ],
    defines = [
        "_FILE_OFFSET_BITS=64",
        "_GNU_SOURCE",
        "BITDEPTH=8",
    ],
    includes = [
        "build",
        "include",
        "include/dav1d",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "include/compat/msvc",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dav1d16",
    srcs = glob(
        [
            "include/dav1d/*.h",
            "include/common/*.h",
            "src/*.c",
            "src/x86/*.h",
            "src/x86/*.c",
            "src/*.h",
        ],
        exclude = [
            "src/x86/msac_init.c",
        ],
    ) + select({
        "@bazel_tools//src/conditions:windows": [
            "include/compat/msvc/stdatomic.h",
            "src/win32/thread.c",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "build/config.h",
        "build/vcs_version.h",
        "build/version.h",
    ],
    copts = [
        "-std=c99",
    ],
    defines = [
        "_FILE_OFFSET_BITS=64",
        "_GNU_SOURCE",
        "BITDEPTH=16",
    ],
    includes = [
        "build",
        "include",
        "include/dav1d",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "include/compat/msvc",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

genrule(
    name = "build_version_h",
    srcs = ["include/dav1d/version.h.in"],
    outs = ["build/version.h"],
    cmd = ("sed " +
           "-e 's/@DAV1D_API_VERSION_MAJOR@/4/g' " +
           "-e 's/@DAV1D_API_VERSION_MINOR@/0/g' " +
           "-e 's/@DAV1D_API_VERSION_PATCH@/0/g' " +
           "$< >$@"),
)

genrule(
    name = "build_vcs_version_h",
    srcs = ["include/vcs_version.h.in"],
    outs = ["build/vcs_version.h"],
    cmd = ("sed " +
           "-e 's/@VCS_TAG@/0.6.0/g' " +
           "$< >$@"),
)

genrule(
    name = "build_config_h",
    outs = ["build/config.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#pragma once",
        "#define ARCH_AARCH64 0",
        "#define ARCH_ARM 0",
        "#define ARCH_PPC64LE 0",
        "#define ARCH_X86 1",
        "#define ARCH_X86_32 0",
        "#define ARCH_X86_64 1",
        "#define CONFIG_16BPC 1",
        "#define CONFIG_8BPC 1",
        "#define CONFIG_LOG 1",
        "#define ENDIANNESS_BIG 0",
        "#define HAVE_ASM 0",
        "",
        "#if defined(_MSC_VER)",
        "",
        "#define HAVE_ALIGNED_MALLOC 1",
        "#define HAVE_IO_H 1",
        "#define STACK_ALIGNMENT 16",
        "#define UNICODE 1",
        "#define _CRT_DECLARE_NONSTDC_NAMES 1 ",
        "#define _UNICODE 1",
        "#define fseeko _fseeki64",
        "#define ftello _ftelli64",
        "",
        "#else",
        "",
        "#define HAVE_CLOCK_GETTIME 1",
        "#define HAVE_DLSYM 1",
        "#define HAVE_POSIX_MEMALIGN 1",
        "#define HAVE_UNISTD_H 1",
        "#define STACK_ALIGNMENT 32",
        "",
        "#endif",
        "EOF",
    ]),
)
