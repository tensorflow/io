# Description:
#   libjpeg-turbo is a drop in replacement for jpeglib optimized with SIMD.

licenses(["notice"])  # custom notice-style license, see LICENSE.md

exports_files(["LICENSE.md"])

cc_library(
    name = "jpeg",
    srcs = [
        "jaricom.c",
        "jcapimin.c",
        "jcapistd.c",
        "jcarith.c",
        "jccoefct.c",
        "jccolor.c",
        "jcdctmgr.c",
        "jchuff.c",
        "jcinit.c",
        "jcmainct.c",
        "jcmarker.c",
        "jcmaster.c",
        "jcomapi.c",
        "jcparam.c",
        "jcphuff.c",
        "jcprepct.c",
        "jcsample.c",
        "jctrans.c",
        "jdapimin.c",
        "jdapistd.c",
        "jdarith.c",
        "jdatadst.c",
        "jdatasrc.c",
        "jdcoefct.c",
        "jdcoefct.h",
        "jdcolor.c",
        "jddctmgr.c",
        "jdhuff.c",
        "jdhuff.h",
        "jdinput.c",
        "jdmainct.c",
        "jdmainct.h",
        "jdmarker.c",
        "jdmaster.c",
        "jdmaster.h",
        "jdmerge.c",
        "jdphuff.c",
        "jdpostct.c",
        "jdsample.c",
        "jdsample.h",
        "jdtrans.c",
        "jerror.c",
        "jfdctflt.c",
        "jfdctfst.c",
        "jfdctint.c",
        "jidctflt.c",
        "jidctfst.c",
        "jidctint.c",
        "jidctred.c",
        "jmemmgr.c",
        "jmemnobs.c",
        "jmemsys.h",
        "jpeg_nbits_table.h",
        "jpegcomp.h",
        "jquant1.c",
        "jquant2.c",
        "jutils.c",
        "jversion.h",
    ],
    hdrs = [
        "jccolext.c",
        "jdcol565.c",
        "jdcolext.c",
        "jdmrg565.c",
        "jdmrgext.c",
        "jstdhuff.c",
    ],
    copts = [],
    includes = [
        "config",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":simd_x86_64",  # ":simd_none"
    ],
)

cc_library(
    name = "simd_none",
    srcs = [
        "jsimd_none.c",
    ],
    hdrs = [],
    copts = [],
    includes = [],
    linkstatic = 1,
    deps = [
        ":header",
    ],
)

cc_library(
    name = "simd_x86_64",
    srcs = [
        "simd/jsimd.h",
        "simd/x86_64/jccolor-avx2.o",
        "simd/x86_64/jccolor-sse2.o",
        "simd/x86_64/jcgray-avx2.o",
        "simd/x86_64/jcgray-sse2.o",
        "simd/x86_64/jchuff-sse2.o",
        "simd/x86_64/jcphuff-sse2.o",
        "simd/x86_64/jcsample-avx2.o",
        "simd/x86_64/jcsample-sse2.o",
        "simd/x86_64/jdcolor-avx2.o",
        "simd/x86_64/jdcolor-sse2.o",
        "simd/x86_64/jdmerge-avx2.o",
        "simd/x86_64/jdmerge-sse2.o",
        "simd/x86_64/jdsample-avx2.o",
        "simd/x86_64/jdsample-sse2.o",
        "simd/x86_64/jfdctflt-sse.o",
        "simd/x86_64/jfdctfst-sse2.o",
        "simd/x86_64/jfdctint-avx2.o",
        "simd/x86_64/jfdctint-sse2.o",
        "simd/x86_64/jidctflt-sse2.o",
        "simd/x86_64/jidctfst-sse2.o",
        "simd/x86_64/jidctint-avx2.o",
        "simd/x86_64/jidctint-sse2.o",
        "simd/x86_64/jidctred-sse2.o",
        "simd/x86_64/jquantf-sse2.o",
        "simd/x86_64/jquanti-avx2.o",
        "simd/x86_64/jquanti-sse2.o",
        "simd/x86_64/jsimd.c",
        "simd/x86_64/jsimdcpu.o",
    ],
    hdrs = [],
    copts = [],
    includes = [
        "simd/x86_64",
    ],
    linkstatic = 1,
    deps = [
        ":header",
    ],
)

assembly_base = (
    "    -I $$(dirname $(location config/jconfig.h))/" +
    "    -I $$(dirname $(location config/jconfigint.h))/" +
    "    -I $$(dirname $(location simd/nasm/jsimdcfg.inc.h))/" +
    "    -I $$(dirname $(location simd/x86_64/jccolext-sse2.asm))/" +
    "    -o $$out" +
    "    $$(dirname $(location simd/x86_64/jccolext-sse2.asm))/$$(basename $${out%.o}.asm)\n"
)

genrule(
    name = "assembly",
    srcs = [
        "config/jconfig.h",
        "config/jconfigint.h",
        "simd/x86_64/jccolext-avx2.asm",
        "simd/x86_64/jccolext-sse2.asm",
        "simd/x86_64/jccolor-avx2.asm",
        "simd/x86_64/jccolor-sse2.asm",
        "simd/x86_64/jcgray-avx2.asm",
        "simd/x86_64/jcgray-sse2.asm",
        "simd/x86_64/jcgryext-avx2.asm",
        "simd/x86_64/jcgryext-sse2.asm",
        "simd/x86_64/jchuff-sse2.asm",
        "simd/x86_64/jcphuff-sse2.asm",
        "simd/x86_64/jcsample-avx2.asm",
        "simd/x86_64/jcsample-sse2.asm",
        "simd/x86_64/jdcolext-avx2.asm",
        "simd/x86_64/jdcolext-sse2.asm",
        "simd/x86_64/jdcolor-avx2.asm",
        "simd/x86_64/jdcolor-sse2.asm",
        "simd/x86_64/jdmerge-avx2.asm",
        "simd/x86_64/jdmerge-sse2.asm",
        "simd/x86_64/jdmrgext-avx2.asm",
        "simd/x86_64/jdmrgext-sse2.asm",
        "simd/x86_64/jdsample-avx2.asm",
        "simd/x86_64/jdsample-sse2.asm",
        "simd/x86_64/jfdctflt-sse.asm",
        "simd/x86_64/jfdctfst-sse2.asm",
        "simd/x86_64/jfdctint-avx2.asm",
        "simd/x86_64/jfdctint-sse2.asm",
        "simd/x86_64/jidctflt-sse2.asm",
        "simd/x86_64/jidctfst-sse2.asm",
        "simd/x86_64/jidctint-avx2.asm",
        "simd/x86_64/jidctint-sse2.asm",
        "simd/x86_64/jidctred-sse2.asm",
        "simd/x86_64/jquantf-sse2.asm",
        "simd/x86_64/jquanti-avx2.asm",
        "simd/x86_64/jquanti-sse2.asm",
        "simd/x86_64/jsimdcpu.asm",
        "simd/nasm/jcolsamp.inc",
        "simd/nasm/jdct.inc",
        "simd/nasm/jpeg_nbits_table.inc",
        "simd/nasm/jsimdcfg.inc",
        "simd/nasm/jsimdcfg.inc.h",
        "simd/nasm/jsimdext.inc",
    ],
    outs = [
        "simd/x86_64/jccolor-avx2.o",
        "simd/x86_64/jccolor-sse2.o",
        "simd/x86_64/jcgray-avx2.o",
        "simd/x86_64/jcgray-sse2.o",
        "simd/x86_64/jchuff-sse2.o",
        "simd/x86_64/jcphuff-sse2.o",
        "simd/x86_64/jcsample-avx2.o",
        "simd/x86_64/jcsample-sse2.o",
        "simd/x86_64/jdcolor-avx2.o",
        "simd/x86_64/jdcolor-sse2.o",
        "simd/x86_64/jdmerge-avx2.o",
        "simd/x86_64/jdmerge-sse2.o",
        "simd/x86_64/jdsample-avx2.o",
        "simd/x86_64/jdsample-sse2.o",
        "simd/x86_64/jfdctflt-sse.o",
        "simd/x86_64/jfdctfst-sse2.o",
        "simd/x86_64/jfdctint-avx2.o",
        "simd/x86_64/jfdctint-sse2.o",
        "simd/x86_64/jidctflt-sse2.o",
        "simd/x86_64/jidctfst-sse2.o",
        "simd/x86_64/jidctint-avx2.o",
        "simd/x86_64/jidctint-sse2.o",
        "simd/x86_64/jidctred-sse2.o",
        "simd/x86_64/jquantf-sse2.o",
        "simd/x86_64/jquanti-avx2.o",
        "simd/x86_64/jquanti-sse2.o",
        "simd/x86_64/jsimdcpu.o",
    ],
    cmd = select({
        "@bazel_tools//src/conditions:windows": (
            "for out in $(OUTS); do\n" +
            "  $(location @nasm//:nasm) " +
            "    -f win64 -DWIN64 -DPIC -D__x86_64__" +
            assembly_base +
            "done"
        ),
        "@bazel_tools//src/conditions:darwin": (
            "for out in $(OUTS); do\n" +
            "  $(location @nasm//:nasm) " +
            "    -f macho64 -DMACHO -DPIC -D__x86_64__" +
            assembly_base +
            "done"
        ),
        "//conditions:default": (
            "for out in $(OUTS); do\n" +
            "  $(location @nasm//:nasm) " +
            "    -f elf64 -DELF -DPIC -D__x86_64__" +
            assembly_base +
            "done"
        ),
    }),
    tools = ["@nasm"],
)

cc_library(
    name = "header",
    srcs = [
        "jchuff.h",
        "jdct.h",
        "jerror.h",
        "jinclude.h",
        "jmorecfg.h",
        "jpegint.h",
        "jpeglib.h",
        "jsimd.h",
        "jsimddct.h",
    ],
    hdrs = [
        "config/jconfig.h",
        "config/jconfigint.h",
    ],
    copts = [],
    includes = [
        "config",
    ],
    visibility = ["//visibility:public"],
)

genrule(
    name = "config_jconfig_h",
    srcs = ["jconfig.h.in"],
    outs = ["config/jconfig.h"],
    cmd = (
        "sed " +
        "-e 's/@JPEG_LIB_VERSION@/62/g' " +
        "-e 's/@VERSION@/2.0.0/g' " +
        "-e 's/@LIBJPEG_TURBO_VERSION_NUMBER@/2000000/g' " +
        "-e 's/#cmakedefine C_ARITH_CODING_SUPPORTED 1/#define C_ARITH_CODING_SUPPORTED 1/g' " +
        "-e 's/#cmakedefine D_ARITH_CODING_SUPPORTED 1/#define D_ARITH_CODING_SUPPORTED 1/g' " +
        "-e 's/#cmakedefine MEM_SRCDST_SUPPORTED 1/#define MEM_SRCDST_SUPPORTED 1/g' " +
        "-e 's/#cmakedefine WITH_SIMD 1/#define WITH_SIMD 1/g' " +
        "-e 's/@BITS_IN_JSAMPLE@/8/g' " +
        "-e 's/#cmakedefine HAVE_LOCALE_H 1/#define HAVE_LOCALE_H 1/g' " +
        "-e 's/#cmakedefine HAVE_STDDEF_H 1/#define HAVE_STDDEF_H 1/g' " +
        "-e 's/#cmakedefine HAVE_STDLIB_H 1/#define HAVE_STDLIB_H 1/g' " +
        "-e 's/#cmakedefine NEED_SYS_TYPES_H 1/#define NEED_SYS_TYPES_H 1/g' " +
        "-e 's/#cmakedefine NEED_BSD_STRINGS 1//g' " +
        "-e 's/#cmakedefine HAVE_UNSIGNED_CHAR 1/#define HAVE_UNSIGNED_CHAR 1/g' " +
        "-e 's/#cmakedefine HAVE_UNSIGNED_SHORT 1/#define HAVE_UNSIGNED_SHORT 1/g' " +
        "-e 's/#cmakedefine INCOMPLETE_TYPES_BROKEN 1//g' " +
        "-e 's/#cmakedefine RIGHT_SHIFT_IS_UNSIGNED 1//g' " +
        "-e 's/#cmakedefine __CHAR_UNSIGNED__ 1//g' " +
        "$< >$@"
    ),
)

genrule(
    name = "config_jconfigint_h",
    srcs = ["jconfigint.h.in"],
    outs = ["config/jconfigint.h"],
    cmd = select({
        "@bazel_tools//src/conditions:windows": (
            "sed " +
            "-e 's/@BUILD@/20180831/g' " +
            "-e 's/@INLINE@/inline/g' " +
            "-e 's/@CMAKE_PROJECT_NAME@/libjpeg-turbo/g' " +
            "-e 's/@VERSION@/2.0.0/g' " +
            "-e 's/@SIZE_T@/8/g' " +
            "-e 's/#cmakedefine HAVE_BUILTIN_CTZL//g' " +
            "-e 's/#cmakedefine HAVE_INTRIN_H//g' " +
            "$< >$@"
        ),
        "//conditions:default": (
            "sed " +
            "-e 's/@BUILD@/20180831/g' " +
            "-e 's/@INLINE@/inline/g' " +
            "-e 's/@CMAKE_PROJECT_NAME@/libjpeg-turbo/g' " +
            "-e 's/@VERSION@/2.0.0/g' " +
            "-e 's/@SIZE_T@/8/g' " +
            "-e 's/#cmakedefine HAVE_BUILTIN_CTZL/#define HAVE_BUILTIN_CTZL/g' " +
            "-e 's/#cmakedefine HAVE_INTRIN_H//g' " +
            "$< >$@"
        ),
    }),
)
