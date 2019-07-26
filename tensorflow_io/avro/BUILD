licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "avro_ops",
    srcs = [
        "kernels/avro_dataset.cc",
        "ops/avro_ops.cc",
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
        ":avro_utils",
    ],
)

# Note: I use gtest directly instead of tf test framework because it
# is not imported into TF io and even if it where it may go away in
# the future, since there is a comment in tensorflow/core/BUILD and
# in there it says that //tensorflow/core:test will be removed from
# the public API
cc_test(
    name = "prefix_tree_test",
    srcs = ["utils/prefix_tree_test.cc"],
    deps = [
        ":avro_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

# TODO(fraudies): Migrate tests to avro c++ implementation
# cc_test(
#     name = "avro_parser_test",
#    srcs = ["utils/avro_parser_test.cc"],
#     deps = [
#         ":avro_utils",
#         "@com_google_googletest//:gtest_main",
#     ],
# )

# TODO(fraudies): For these tests we need to figure out how to get
# the test dependencies from TF included
# Tried to define tf_cc_test target which does that for the internal
# build but w/out success
#cc_test(
#    name = "value_buffer_test",
#    srcs = ["utils/value_buffer_test.cc"],
#    deps = [
#        ":avro_utils",
#        "//tensorflow/core:testlib",
#        "@com_google_googletest//:gtest_main",
#    ],
#)
#cc_test(
#    name = "avro_parser_tree_test",
#    srcs = ["utils/avro_parser_tree_test.cc"],
#    deps = [
#        ":avro_utils",
#        "//tensorflow/core:testlib",
#        "@com_google_googletest//:gtest_main",
#    ],
#)

cc_library(
    name = "avro_utils_api",
    hdrs = ["utils/prefix_tree.h",
            "utils/value_buffer.h",
            "utils/avro_parser.h",
            "utils/avro_parser_tree.h",
            "utils/parse_avro_attrs.h",
            "utils/avro_file_stream_reader.h",
            ],
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "avro_utils",
    srcs = ["utils/prefix_tree.cc",
            "utils/value_buffer.cc",
            "utils/avro_parser.cc",
            "utils/avro_parser_tree.cc",
            "utils/parse_avro_attrs.cc",
            "utils/avro_file_stream_reader.cc",
            ],
    deps = [
        ":avro_utils_api",
        "@avro",
    ],
)