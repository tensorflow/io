# Description:
#   GeoTIFF library

licenses(["notice"])  # Public/BSD-like licence

exports_files(["LICENSE"])

cc_library(
    name = "libgeotiff",
    srcs = glob([
        "libxtiff/*.c",
        "*.c",
        "*.inc",
    ]),
    hdrs = glob([
        "libxtiff/*.h",
        "*.h",
    ]) + [
        "geo_config.h",
    ],
    defines = [],
    includes = [
        "libxtiff",
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "@libtiff",
        "@proj",
    ],
)

genrule(
    name = "geo_config_h",
    outs = ["geo_config.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#ifndef GEO_CONFIG_H",
        "#define GEO_CONFIG_H",
        "#define STDC_HEADERS 1",
        "#define HAVE_STDLIB_H 1",
        "#define HAVE_STRING_H 1",
        "#define HAVE_STRINGS_H 1",
        "#define HAVE_LIBPROJ 1",
        "#define HAVE_PROJECTS_H 1",
        "/* #undef GEO_NORMALIZE_DISABLE_TOWGS84 */",
        "#endif /* ndef GEO_CONFIG_H */",
        "EOF",
    ]),
)
