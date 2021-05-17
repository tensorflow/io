# Description:
#   dav1d library

licenses(["notice"])  # BSD license

exports_files(["COPYING"])

cc_library(
    name = "dav1d8",
    srcs = [
        "include8/common/bitdepth.h",
        "include8/common/dump.h",
        "src/cdef_apply_tmpl.c",
        "src/cdef_tmpl.c",
        "src/fg_apply_tmpl.c",
        "src/film_grain_tmpl.c",
        "src/ipred_prepare_tmpl.c",
        "src/ipred_tmpl.c",
        "src/itx_tmpl.c",
        "src/lf_apply_tmpl.c",
        "src/loopfilter_tmpl.c",
        "src/looprestoration_tmpl.c",
        "src/lr_apply_tmpl.c",
        "src/mc_tmpl.c",
        "src/recon_tmpl.c",
    ],
    hdrs = [],
    copts = [],
    defines = [],
    includes = [
        "include8",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dav1d16",
    srcs = [
        "include16/common/bitdepth.h",
        "include16/common/dump.h",
        "src/cdef_apply_tmpl.c",
        "src/cdef_tmpl.c",
        "src/fg_apply_tmpl.c",
        "src/film_grain_tmpl.c",
        "src/ipred_prepare_tmpl.c",
        "src/ipred_tmpl.c",
        "src/itx_tmpl.c",
        "src/lf_apply_tmpl.c",
        "src/loopfilter_tmpl.c",
        "src/looprestoration_tmpl.c",
        "src/lr_apply_tmpl.c",
        "src/mc_tmpl.c",
        "src/recon_tmpl.c",
    ],
    hdrs = [],
    copts = [],
    defines = [],
    includes = [
        "include16",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
    ],
    alwayslink = 1,
)

cc_library(
    name = "source",
    srcs = [
        "include/common/bitdepth.h",
        "src/cdf.c",
        "src/cpu.c",
        "src/data.c",
        "src/decode.c",
        "src/dequant_tables.c",
        "src/getbits.c",
        "src/intra_edge.c",
        "src/itx_1d.c",
        "src/lf_mask.c",
        "src/lib.c",
        "src/log.c",
        "src/msac.c",
        "src/obu.c",
        "src/picture.c",
        "src/qm.c",
        "src/ref.c",
        "src/ref_mvs.c",
        "src/scan.c",
        "src/tables.c",
        "src/thread_task.c",
        "src/warpmv.c",
        "src/wedge.c",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "src/win32/thread.c",
        ],
        "//conditions:default": [],
    }),
    hdrs = [],
    copts = [],
    defines = [],
    visibility = ["//visibility:public"],
    deps = [
        ":header",
    ],
)

cc_library(
    name = "header",
    srcs = [],
    hdrs = [
        "build/config.h",
        "build/vcs_version.h",
        "build/version.h",
        "include/common/attributes.h",
        "include/common/intops.h",
        "include/common/mem.h",
        "include/common/validate.h",
    ] + glob([
        "include/dav1d/*.h",
        "src/*.h",
        "src/x86/*.h",
    ]) + select({
        "@bazel_tools//src/conditions:windows": [
            "include/compat/msvc/stdatomic.h",
        ],
        "//conditions:default": [],
    }),
    copts = [
        "-std=c99",
    ],
    defines = [
        "DAV1D_API=",
        "_FILE_OFFSET_BITS=64",
        "_GNU_SOURCE",
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
    deps = [],
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
