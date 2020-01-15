workspace(name = "org_tensorflow_io")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

load("//third_party/toolchains/tf:tf_configure.bzl", "tf_configure")
tf_configure(name = "local_config_tf")

load("//third_party/toolchains/gpu:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "a31397714a353587413d307337d0b58f8a2e20e2b9d02f2e24e3463fa4eeda81",
    strip_prefix = "re2-2018-10-01",
    urls = [
        "https://github.com/google/re2/archive/2018-10-01.tar.gz",
    ],
)
