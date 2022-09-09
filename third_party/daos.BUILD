package(default_visibility = ["//visibility:public"])

cc_library(
    name = "daos",
    hdrs = glob(
        [
            "src/include/**/*.h",
        ],
    ) + [
        "src/include/daos_version.h",
    ],
    copts = [],
    includes = ["src/include"],
    deps = [
        "@util_linux//:uuid",
    ],
)

genrule(
    name = "daos_version_h",
    srcs = [
        "src/include/daos_version.h.in",
    ],
    outs = [
        "src/include/daos_version.h",
    ],
    cmd = ("sed " +
           "-e 's/@TMPL_MAJOR@/2/g' " +
           "-e 's/@TMPL_MINOR@/0/g' " +
           "-e 's/@TMPL_FIX@/2/g' " +
           "$< >$@"),
)
