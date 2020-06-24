# Description:
#   libjpeg-turbo is a drop in replacement for jpeglib optimized with SIMD.

licenses(["notice"])  # custom notice-style license, see LICENSE.md

exports_files(["LICENSE.md"])

# TODO: add simd support

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
        "jchuff.h",
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
        "jdct.h",
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
        "jerror.h",
        "jfdctflt.c",
        "jfdctfst.c",
        "jfdctint.c",
        "jidctflt.c",
        "jidctfst.c",
        "jidctint.c",
        "jidctred.c",
        "jinclude.h",
        "jmemmgr.c",
        "jmemnobs.c",
        "jmemsys.h",
        "jmorecfg.h",
        "jpeg_nbits_table.h",
        "jpegcomp.h",
        "jpegint.h",
        "jpeglib.h",
        "jquant1.c",
        "jquant2.c",
        "jsimd.h",
        "jsimd_none.c",
        "jsimddct.h",
        "jutils.c",
        "jversion.h",
    ],
    hdrs = [
        "config/jconfig.h",
        "config/jconfigint.h",
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
        "-e 's/cmakedefine C_ARITH_CODING_SUPPORTED 1/define C_ARITH_CODING_SUPPORTED 1/g' " +
        "-e 's/cmakedefine D_ARITH_CODING_SUPPORTED 1/define D_ARITH_CODING_SUPPORTED 1/g' " +
        "-e 's/cmakedefine MEM_SRCDST_SUPPORTED 1/define MEM_SRCDST_SUPPORTED 1/g' " +
        "-e 's/cmakedefine WITH_SIMD 1//g' " +
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
