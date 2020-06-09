# Description:
#   libgav1 decoder for AVIF library

licenses(["notice"])  # Apache license

exports_files(["LICENSE"])

cc_library(
    name = "libgav1",
    srcs = glob(
        [
            "src/**/*.cc",
            "src/**/*.h",
        ],
    ),
    hdrs = glob([
        "src/**/*.inc",
    ]),
    defines = [
        "LIBGAV1_MAX_BITDEPTH=8",
    ],
    includes = [
        "src",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)
