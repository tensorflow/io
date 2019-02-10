workspace(name = "org_tensorflow_io")

load("//third_party/tf:tf_configure.bzl", "tf_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

tf_configure(
    name = "local_config_tf",
)

http_archive(
    name = "boringssl",
    sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
    strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
    urls = [
        "https://mirror.bazel.build/github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
        "https://github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
    ],
)

http_archive(
    name = "zlib",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "https://mirror.bazel.build/zlib.net/zlib-1.2.11.tar.gz",
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
    build_file = "//third_party:zlib.BUILD",
)

http_archive(
    name = "curl",
    sha256 = "e9c37986337743f37fd14fe8737f246e97aec94b39d1b71e8a5973f72a9fc4f5",
    strip_prefix = "curl-7.60.0",
    urls = [
        "https://mirror.bazel.build/curl.haxx.se/download/curl-7.60.0.tar.gz",
        "https://curl.haxx.se/download/curl-7.60.0.tar.gz",
    ],
    build_file = "//third_party:curl.BUILD",
)

http_archive(
    name = "kafka",
    sha256 = "cc6ebbcd0a826eec1b8ce1f625ffe71b53ef3290f8192b6cae38412a958f4fd3",
    strip_prefix = "librdkafka-0.11.5",
    urls = [
        "https://mirror.bazel.build/github.com/edenhill/librdkafka/archive/v0.11.5.tar.gz",
        "https://github.com/edenhill/librdkafka/archive/v0.11.5.tar.gz",
    ],
    build_file = "//third_party:kafka.BUILD",
    patches = [
        "//third_party:kafka.patch",
    ],
)

http_archive(
    name = "aws",
    urls = [
        "https://mirror.bazel.build/github.com/aws/aws-sdk-cpp/archive/1.3.15.tar.gz",
        "https://github.com/aws/aws-sdk-cpp/archive/1.3.15.tar.gz",
    ],
    sha256 = "b888d8ce5fc10254c3dd6c9020c7764dd53cf39cf011249d0b4deda895de1b7c",
    strip_prefix = "aws-sdk-cpp-1.3.15",
    build_file = "//third_party:aws.BUILD",
)

http_archive(
    name = "snappy",
    urls = [
        "https://mirror.bazel.build/github.com/google/snappy/archive/1.1.7.tar.gz",
        "https://github.com/google/snappy/archive/1.1.7.tar.gz",
    ],
    sha256 = "3dfa02e873ff51a11ee02b9ca391807f0c8ea0529a4924afa645fbf97163f9d4",
    strip_prefix = "snappy-1.1.7",
    build_file = "//third_party:snappy.BUILD",
)

# Parquet needs generated parquet_types.h and parquet_types.cpp which are generated
# from src/parquet/parquet.thrift in apache-parquet-cpp-1.4.0.tar.gz.
#
# Generating parquet_types.h and parquet_types.cpp, however, needs both bison and flex
# installed, which is really an unnecessary step.
#
# We use the following step to generate the parquet_types.h and parquet_types.cpp files:
#  - In third_party directory, run `docker run -i -t --rm -v $PWD:/v -w /v ubuntu:16.04 bash -x /v/parquet.header`
#  - Once complete, a parquet.patch file will be generated which could be used as a patch in bazel
#
# $ cd third_party
# $ docker run -i -t --rm -v $PWD:/v -w /v ubuntu:16.04 bash -x /v/parquet.header
http_archive(
    name = "arrow",
    urls = [
        "https://mirror.bazel.build/github.com/apache/arrow/archive/apache-arrow-0.11.1.tar.gz",
        "https://github.com/apache/arrow/archive/apache-arrow-0.11.1.tar.gz",
    ],
    sha256 = "3219c4e87e7cf979017f0cc5bc5dd6a3611d0fc750e821911fab998599dc125b",
    strip_prefix = "arrow-apache-arrow-0.11.1",
    build_file = "//third_party:arrow.BUILD",
    patches = [
        "//third_party:parquet.patch",
    ],
    patch_args = ["-p1"],
)

http_archive(
    name = "boost",
    urls = [
        "https://mirror.bazel.build/dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz",
        "https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz"
    ],
    sha256 = "8aa4e330c870ef50a896634c931adf468b21f8a69b77007e45c444151229f665",
    strip_prefix = "boost_1_67_0",
    build_file = "//third_party:boost.BUILD",
)

http_archive(
    name = "thrift",
    urls = [
        "https://mirror.bazel.build/github.com/apache/thrift/archive/0.11.0.tar.gz",
        "https://github.com/apache/thrift/archive/0.11.0.tar.gz",
    ],
    sha256 = "0e324569321a1b626381baabbb98000c8dd3a59697292dbcc71e67135af0fefd",
    strip_prefix = "thrift-0.11.0",
    build_file = "//third_party:thrift.BUILD",
)

http_archive(
    name = "libwebp",
    urls = [
        "https://mirror.bazel.build/github.com/webmproject/libwebp/archive/v1.0.1.tar.gz",
        "https://github.com/webmproject/libwebp/archive/v1.0.1.tar.gz",
    ],
    sha256 = "c2838544d4898a4bbb6c1d113e0aa50c4bdfc886df0dcfbfa5c42e788cb6f382",
    strip_prefix = "libwebp-1.0.1",
    build_file = "//third_party:libwebp.BUILD",
)

http_archive(
    name = "com_github_google_flatbuffers",
    urls = [
        "https://mirror.bazel.build/github.com/google/flatbuffers/archive/b99332efd732e6faf60bb7ce1ce5902ed65d5ba3.tar.gz",
        "https://github.com/google/flatbuffers/archive/b99332efd732e6faf60bb7ce1ce5902ed65d5ba3.tar.gz",
    ],
    sha256 = "12a13686cab7ffaf8ea01711b8f55e1dbd3bf059b7c46a25fefa1250bdd9dd23",
    strip_prefix = "flatbuffers-b99332efd732e6faf60bb7ce1ce5902ed65d5ba3",
)

http_archive(
    name = "ffmpeg_2_8",
    urls = [
        "https://mirror.bazel.build/github.com/FFmpeg/FFmpeg/archive/n2.8.15.tar.gz",
        "https://github.com/FFmpeg/FFmpeg/archive/n2.8.15.tar.gz",
    ],
    sha256 = "8ba1b91a14431fe37091936c3a34469d7473965ab9edde0343c88f2d920bd918",
    strip_prefix = "FFmpeg-n2.8.15",
    build_file = "//third_party:ffmpeg_2_8.BUILD",
)

http_archive(
    name = "ffmpeg_3_4",
    urls = [
        "https://mirror.bazel.build/github.com/FFmpeg/FFmpeg/archive/n3.4.4.tar.gz",
        "https://github.com/FFmpeg/FFmpeg/archive/n3.4.4.tar.gz",
    ],
    sha256 = "bbccc87cd031498728bcc2dba5596a47e6fd92b2cec060a71feef65617a261fe",
    strip_prefix = "FFmpeg-n3.4.4",
    build_file = "//third_party:ffmpeg_3_4.BUILD",
)

http_archive(
    name = "libav_9_20",
    urls = [
        "https://mirror.bazel.build/github.com/libav/libav/archive/v9.20.tar.gz",
        "https://github.com/libav/libav/archive/v9.20.tar.gz",
    ],
    sha256 = "ecc2389bc857602450196c9240e1ebc59066980f5d42e977efe0f498145775d4",
    strip_prefix = "libav-9.20",
    build_file = "//third_party:libav_9_20.BUILD",
)

http_archive(
    name = "lmdb",
    urls = [
        "https://mirror.bazel.build/github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
        "https://github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
    ],
    sha256 = "f3927859882eb608868c8c31586bb7eb84562a40a6bf5cc3e13b6b564641ea28",
    strip_prefix = "lmdb-LMDB_0.9.22/libraries/liblmdb",
    build_file = "//third_party:lmdb.BUILD",
)
http_archive(
    name = "libtiff",
    urls = [
        "https://mirror.bazel.build/download.osgeo.org/libtiff/tiff-4.0.10.tar.gz",
        "https://download.osgeo.org/libtiff/tiff-4.0.10.tar.gz",
    ],
    sha256 = "2c52d11ccaf767457db0c46795d9c7d1a8d8f76f68b0b800a3dfe45786b996e4",
    strip_prefix = "tiff-4.0.10",
    build_file = "//third_party:libtiff.BUILD",
)
