# Description:
#   Proj library

licenses(["notice"])  # Public/BSD-like licence

exports_files(["LICENSE"])

cc_library(
    name = "sqlite",
    srcs = [
        "sqlite3.c",
    ],
    hdrs = [
        "sqlite3.h",
    ],
    defines = [],
    includes = [
        ".",
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

genrule(
    name = "proj_config_h",
    outs = ["proj_config.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define HAVE_CXX11 1",
        "#define HAVE_DLFCN_H 1",
        "#define HAVE_GCC_WARNING_ZERO_AS_NULL_POINTER_CONSTANT 1",
        "#define HAVE_INTTYPES_H 1",
        "/* #undef HAVE_JNI_H */",
        "#define HAVE_LIBM 1",
        "#define HAVE_LIBPTHREAD 1",
        "#define HAVE_LOCALECONV 1",
        "#define HAVE_MEMORY_H 1",
        "#define HAVE_PTHREAD_MUTEX_RECURSIVE /**/",
        "#define HAVE_STDINT_H 1",
        "#define HAVE_STDLIB_H 1",
        "#define HAVE_STRERROR 1",
        "#define HAVE_STRINGS_H 1",
        "#define HAVE_STRING_H 1",
        "#define HAVE_SYS_STAT_H 1",
        "#define HAVE_SYS_TYPES_H 1",
        "#define HAVE_UNISTD_H 1",
        "/* #undef JNI_ENABLED */",
        "#define LT_OBJDIR \".libs/\"",
        "#define PACKAGE \"proj\"",
        "#define PACKAGE_BUGREPORT \"https://github.com/OSGeo/PROJ/issues\"",
        "#define PACKAGE_NAME \"PROJ\"",
        "#define PACKAGE_STRING \"PROJ 6.2.0\"",
        "#define PACKAGE_TARNAME \"proj\"",
        "#define PACKAGE_URL \"https://proj.org\"",
        "#define PACKAGE_VERSION \"6.2.0\"",
        "#define STDC_HEADERS 1",
        "#define VERSION \"6.2.0\"",
        "EOF",
    ]),
)
