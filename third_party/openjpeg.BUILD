# Description:
#   OpenJPEG library

licenses(["notice"])  # 2-clauses BSD license

exports_files(["LICENSE"])

cc_library(
    name = "openjpeg",
    srcs = glob(
        [
            "src/lib/openjp2/*.c",
            "src/lib/openjp2/*.h",
        ],
        exclude = [
            "src/lib/openjp2/*_manager.c",
            "src/lib/openjp2/*_manager.h",
        ],
    ),
    hdrs = [
        "config/opj_config.h",
        "config/opj_config_private.h",
    ],
    defines = [
        "OPJ_STATIC",
    ],
    includes = [
        "config",
        "src/lib/openjp2",
    ],
    visibility = ["//visibility:public"],
)

genrule(
    name = "opj_config_private_h",
    outs = ["config/opj_config_private.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define OPJ_HAVE_INTTYPES_H 1",
        '#define OPJ_PACKAGE_VERSION "2.3.1"',
        "",
        "#if !defined(_MSC_VER)",
        "",
        "#define OPJ_HAVE_FSEEKO ON",
        "#define OPJ_HAVE_MEMALIGN",
        "#define OPJ_HAVE_POSIX_MEMALIGN",
        "#if !defined(_POSIX_C_SOURCE)",
        "#if defined(OPJ_HAVE_FSEEKO) || defined(OPJ_HAVE_POSIX_MEMALIGN)",
        "#define _POSIX_C_SOURCE 200112L",
        "#endif",
        "#endif",
        "",
        "#endif",
        "EOF",
    ]),
)

genrule(
    name = "opj_config_h",
    outs = ["config/opj_config.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define OPJ_HAVE_STDINT_H 1",
        "#define OPJ_VERSION_MAJOR 2",
        "#define OPJ_VERSION_MINOR 3",
        "#define OPJ_VERSION_BUILD 1",
        "EOF",
    ]),
)
