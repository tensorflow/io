package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD license

cc_library(
    name = "vorbis",
    srcs = glob(
        [
            "lib/**/*.h",
            "lib/**/*.c",
        ],
        exclude = [
            "lib/barkmel.c",
            "lib/psytune.c",
            "lib/tone.c",
            "lib/misc.c",
        ],
    ) + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "lib/misc.c",
        ],
    }),
    hdrs = glob([
        "include/vorbis/*.h",
    ]),
    copts = [],
    includes = [
        "include",
        "lib",
    ],
    deps = [
        "@ogg",
    ],
)
