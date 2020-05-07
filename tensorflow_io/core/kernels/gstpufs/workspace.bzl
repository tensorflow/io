"""loads the libmemcached library, used by TFIO."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def gstpufs_repositories():
    http_archive(
        name = "libmemcached",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/launchpad.net/libmemcached/1.0/1.0.18/+download/libmemcached-1.0.18.tar.gz",
            "https://launchpad.net/libmemcached/1.0/1.0.18/+download/libmemcached-1.0.18.tar.gz",
        ],
        sha256 = "e22c0bb032fde08f53de9ffbc5a128233041d9f33b5de022c0978a2149885f82",
        strip_prefix = "libmemcached-1.0.18",
        build_file = "//third_party:libmemcached.BUILD",
    )
