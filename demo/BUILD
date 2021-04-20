load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "hello-time",
    srcs = ["hello-time.cc"],
    visibility = ["//visibility:public"],
    linkopts = [
        "-lcrypto",
        "-pthread",
        "-L/usr/local/lib/",
        "-lsasl2",
    ],
    deps = [
      "@liborc",
    ]
)