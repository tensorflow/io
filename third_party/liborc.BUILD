load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "liborc",
    cmake_options = [
        "-DBUILD_JAVA=OFF",
        "-DBUILD_CPP_TESTS=OFF",
    ],
    lib_source = "@liborc//:all_srcs",
    out_include_dir = "include",
    out_static_libs = [
        "liborc.a",
        "libzstd.a",
        "libhdfspp_static.a",
    ],
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
)
