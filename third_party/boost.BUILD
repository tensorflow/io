# Description:
#   Boost C++ Library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Boost Software License

exports_files(["LICENSE_1_0.txt"])

cc_library(
    name = "boost",
    srcs = glob([
        "boost/**/*.hpp",
        "boost/predef/**/*.h",
        "boost/detail/**/*.ipp",
        "boost/asio/**/*.ipp",
        "boost/date_time/**/*.ipp",
        "boost/xpressive/detail/**/*.ipp",
    ]) + glob([
        "libs/filesystem/src/*.cpp",
        "libs/iostreams/src/*.cpp",
        "libs/regex/src/*.cpp",
        "libs/system/src/*.cpp",
    ]) + [
        "boost/predef.h",
        "libs/filesystem/src/error_handling.hpp",
        "libs/regex/src/internals.hpp",
    ],
    defines = [
        "BOOST_ALL_NO_LIB=1",
    ],
    includes = [
        ".",
    ],
    deps = [
        "@bzip2",
        "@xz//:lzma",
        "@zlib",
        "@zstd",
    ],
)
