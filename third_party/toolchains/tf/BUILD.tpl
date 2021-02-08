package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tf_c_header_lib",
    hdrs = [":tf_c_header_include"],
    include_prefix = "tensorflow/c",
    strip_include_prefix = "include_c",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtensorflow_framework",
    srcs = [":libtensorflow_framework.so"],
    #data = ["lib/libtensorflow_framework.so"],
    visibility = ["//visibility:public"],
)

%{TF_HEADER_GENRULE}
%{TF_C_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}
