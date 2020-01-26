package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD license

exports_files(["COPYRIGHT"])

cc_library(
    name = "postgresql",
    srcs = glob([
        "src/include/*.h",
"src/interfaces/libpq/*.c",
    ]),
    hdrs = [
        "config/pg_config_ext.h",
        "src/interfaces/libpq/libpq-fe.h",
    ],
    copts = [
    ],
    includes = [
        "config",
        "src/include",
        "src/interfaces/libpq",
    ],
)

genrule(
    name = "pg_config_ext_h",
    outs = ["config/pg_config_ext.h"],
    cmd = select({
        "@bazel_tools//src/conditions:windows": "\n".join([
            "cat <<'EOF' >$@",
            "#define PG_INT64_TYPE long long int",
            "EOF",
        ]),
        "//conditions:default": "\n".join([
            "cat <<'EOF' >$@",
            "#define PG_INT64_TYPE long int",
            "EOF",
        ]),
    }),
)
