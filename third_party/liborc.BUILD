load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "liborc",
    lib_source = "@liborc//:all_srcs",
    cmake_options = [
      "-DBUILD_JAVA=OFF",
      "-DBUILD_TESTING=OFF",
      "-DBUILD_CPP_TESTS=OFF",
      "-DSTOP_BUILD_ON_WARNING=OFF",
      # "-Werror",
      # "-DCMAKE_BUILD_TYPE=DEBUG"
    ],
    visibility = ["//visibility:public"],
    out_include_dir = "include",
    out_static_libs = [
      "liborc.a",
      "libzstd.a",
      "libhdfspp_static.a",
    ],
    tags = ["requires-network"],
)