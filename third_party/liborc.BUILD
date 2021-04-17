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
      "-DCMAKE_BUILD_TYPE=DEBUG"
    ],
    visibility = ["//visibility:public"],
    out_include_dir = "include",
    out_static_libs = [
      "liborc.a",
      "libprotoc.a",
      "libz.a",
      "liblz4.a",
      "libprotobuf.a",
      "libsnappy.a",
      "libzstd.a",
      "libhdfspp_static.a",
    ],
    tags = ["requires-network"],
)