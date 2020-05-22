package(default_visibility = ["//visibility:public"])

exports_files(["libmemcached.mem_config.h"])

cc_library(
    name = "openexr",
    srcs = [
        "openexr/b44ExpLogTable.h",
        "openexr/dwaLookups.h",
        "openexr/eLut.h",
        "openexr/toFloat.h",
    ],
    hdrs = [],
    copts = [],
    includes = ["openexr"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "font",
    srcs = [
        "font/font_opensans_regular.h",
    ],
    hdrs = [],
    copts = [],
    includes = ["font"],
    visibility = ["//visibility:public"],
    deps = [],
)

# parquet_types.[h|cpp] are generated from apache arrow 0.16.0
# on Ubuntu 16.04 with default gcc/g++/cmake/flex/bison.
cc_library(
    name = "parquet",
    srcs = [
        "parquet/parquet_types.h",
    ],
    hdrs = [],
    copts = [],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [],
)

exports_files([
    "parquet/parquet_types.cpp",
])
