# Description:
#   Avro Library

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

cc_library(
    name = "avro",
    srcs = glob(
        [
            "api/**/*.hh",
            "impl/**/*.hh",
            "impl/**/*.cc",
        ],
        exclude = [
            "impl/avrogencpp.cc",
        ],
    ),
    hdrs = [],
    copts = [],
    defines = [
        "SNAPPY_CODEC_AVAILABLE",
    ],
    includes = [
        "api",
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "@boost",
        "@snappy",
    ],
)
