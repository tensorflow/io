load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

licenses(["notice"])

cc_library(
    name = "memcached_dao_interfaces",
    hdrs = [
        "memcached_dao_interface.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        "@libmemcached",
    ],
)

cc_library(
    name = "memcached_file_block_cache",
    srcs = ["memcached_file_block_cache.cc"],
    hdrs = ["memcached_file_block_cache.h"],
    copts = tf_io_copts(),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":memcached_dao_interfaces",
        "@com_google_absl//absl/container:flat_hash_map",
        "@farmhash_archive//:farmhash",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "gce_memcached_server_list_provider",
    srcs = [
        "gce_memcached_server_list_provider.cc",
    ],
    hdrs = [
        "gce_memcached_server_list_provider.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/memory",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "memcached_file_system",
    srcs = ["memcached_file_system.cc"],
    hdrs = ["memcached_file_system.h"],
    copts = tf_io_copts(),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":gce_memcached_server_list_provider",
        ":memcached_file_block_cache",
        "@com_google_absl//absl/memory",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "gs_memcached_file_system",
    srcs = ["gs_memcached_file_system.cc"],
    hdrs = ["gs_memcached_file_system.h"],
    copts = tf_io_copts(),
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":gce_memcached_server_list_provider",
        ":memcached_file_system",
        "@com_google_absl//absl/memory",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
