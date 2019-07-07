licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

c_opts = [
    "-pthread",
    "-std=c++11",
    "-DNDEBUG",
]

cc_binary(
    name = "_azfs.so",
    srcs = [],
    copts = c_opts,
    linkshared = True,
    deps = [
        ":azfs",
    ],
)

cc_library(
    name = "azfs",
    srcs = [
        "azfs/azfs.cc",
        "azfs/azfs.h",
        "azfs/azfs_ops.cc",
    ],
    copts = c_opts,
    deps = [
        ":azfs_random_access_file",
        ":azfs_readonly_memory_region",
        ":azfs_writable_file",
    ],
)

cc_library(
    name = "azfs_client",
    srcs = [
        "azfs/azfs_client.cc",
    ],
    hdrs = [
        "azfs/azfs_client.h",
    ],
    copts = c_opts,
    deps = [
        "@com_github_azure_azure_storage_cpplite//:azure",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "azfs_writable_file",
    srcs = [
        "azfs/azfs_writable_file.cc",
    ],
    hdrs = [
        "azfs/azfs_writable_file.h",
    ],
    copts = c_opts,
    deps = [
        ":azfs_client",
    ],
)

cc_library(
    name = "azfs_random_access_file",
    srcs = [
        "azfs/azfs_random_access_file.cc",
    ],
    hdrs = [
        "azfs/azfs_random_access_file.h",
    ],
    copts = c_opts,
    deps = [
        ":azfs_client",
    ],
)

cc_library(
    name = "azfs_readonly_memory_region",
    srcs = [
        "azfs/azfs_readonly_memory_region.h",
    ],
    copts = c_opts,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

# bazel test \
#   --action_env=TF_AZURE_USE_DEV_STORAGE=1 \
#   //tensorflow_io/azure:azfs_cc_test
cc_test(
    name = "azfs_cc_test",
    srcs = ["azfs/azfs_test.cc"],
    deps = [
        ":azfs",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "azfs_ops_py",
    srcs = [
        "azfs_ops.py",
    ],
    data = [
        ":_azfs.so",
    ],
    srcs_version = "PY2AND3",
)
