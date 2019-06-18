licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "avro_ops",
    srcs = [
        "ops/avro_record_dataset.cc",
        "ops/parse_avro_record.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        ":avro_c",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "avro_c",
    srcs = ["avro_c/lib/libavro.so"],
    includes = ["avro_c/include/"],
    visibility = ["//visibility:public"],
)
