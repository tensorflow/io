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
    name = "com_github_madler_zlib",
    build_file = "//third_party:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "https://mirror.bazel.build/zlib.net/zlib-1.2.11.tar.gz",
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
)

http_archive(
    name = "curl",
    build_file = "//third_party:curl.BUILD",
    sha256 = "e9c37986337743f37fd14fe8737f246e97aec94b39d1b71e8a5973f72a9fc4f5",
    strip_prefix = "curl-7.60.0",
    urls = [
        "https://mirror.bazel.build/curl.haxx.se/download/curl-7.60.0.tar.gz",
        "https://curl.haxx.se/download/curl-7.60.0.tar.gz",
    ],
)

http_archive(
    name = "kafka",
    build_file = "//third_party:kafka.BUILD",
    patches = [
        "//third_party:kafka.patch",
    ],
    sha256 = "9c0afb8b53779d968225edf1e79da48a162895ad557900f75e7978f65e642032",
    strip_prefix = "librdkafka-0.11.6",
    urls = [
        "https://mirror.bazel.build/github.com/edenhill/librdkafka/archive/v0.11.6.tar.gz",
        "https://github.com/edenhill/librdkafka/archive/v0.11.6.tar.gz",
    ],
)

http_archive(
    name = "aws",
    build_file = "//third_party:aws.BUILD",
    sha256 = "b888d8ce5fc10254c3dd6c9020c7764dd53cf39cf011249d0b4deda895de1b7c",
    strip_prefix = "aws-sdk-cpp-1.3.15",
    urls = [
        "https://mirror.bazel.build/github.com/aws/aws-sdk-cpp/archive/1.3.15.tar.gz",
        "https://github.com/aws/aws-sdk-cpp/archive/1.3.15.tar.gz",
    ],
)

http_archive(
    name = "snappy",
    build_file = "//third_party:snappy.BUILD",
    sha256 = "3dfa02e873ff51a11ee02b9ca391807f0c8ea0529a4924afa645fbf97163f9d4",
    strip_prefix = "snappy-1.1.7",
    urls = [
        "https://mirror.bazel.build/github.com/google/snappy/archive/1.1.7.tar.gz",
        "https://github.com/google/snappy/archive/1.1.7.tar.gz",
    ],
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
    build_file = "//third_party:arrow.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:parquet.patch",
    ],
    sha256 = "3219c4e87e7cf979017f0cc5bc5dd6a3611d0fc750e821911fab998599dc125b",
    strip_prefix = "arrow-apache-arrow-0.11.1",
    urls = [
        "https://mirror.bazel.build/github.com/apache/arrow/archive/apache-arrow-0.11.1.tar.gz",
        "https://github.com/apache/arrow/archive/apache-arrow-0.11.1.tar.gz",
    ],
)

http_archive(
    name = "boost",
    build_file = "//third_party:boost.BUILD",
    sha256 = "8aa4e330c870ef50a896634c931adf468b21f8a69b77007e45c444151229f665",
    strip_prefix = "boost_1_67_0",
    urls = [
        "https://mirror.bazel.build/dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz",
        "https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz",
    ],
)

http_archive(
    name = "thrift",
    build_file = "//third_party:thrift.BUILD",
    sha256 = "0e324569321a1b626381baabbb98000c8dd3a59697292dbcc71e67135af0fefd",
    strip_prefix = "thrift-0.11.0",
    urls = [
        "https://mirror.bazel.build/github.com/apache/thrift/archive/0.11.0.tar.gz",
        "https://github.com/apache/thrift/archive/0.11.0.tar.gz",
    ],
)

http_archive(
    name = "libwebp",
    build_file = "//third_party:libwebp.BUILD",
    sha256 = "c2838544d4898a4bbb6c1d113e0aa50c4bdfc886df0dcfbfa5c42e788cb6f382",
    strip_prefix = "libwebp-1.0.1",
    urls = [
        "https://mirror.bazel.build/github.com/webmproject/libwebp/archive/v1.0.1.tar.gz",
        "https://github.com/webmproject/libwebp/archive/v1.0.1.tar.gz",
    ],
)

http_archive(
    name = "com_github_google_flatbuffers",
    sha256 = "12a13686cab7ffaf8ea01711b8f55e1dbd3bf059b7c46a25fefa1250bdd9dd23",
    strip_prefix = "flatbuffers-b99332efd732e6faf60bb7ce1ce5902ed65d5ba3",
    urls = [
        "https://mirror.bazel.build/github.com/google/flatbuffers/archive/b99332efd732e6faf60bb7ce1ce5902ed65d5ba3.tar.gz",
        "https://github.com/google/flatbuffers/archive/b99332efd732e6faf60bb7ce1ce5902ed65d5ba3.tar.gz",
    ],
)

http_archive(
    name = "ffmpeg_2_8",
    build_file = "//third_party:ffmpeg_2_8.BUILD",
    sha256 = "8ba1b91a14431fe37091936c3a34469d7473965ab9edde0343c88f2d920bd918",
    strip_prefix = "FFmpeg-n2.8.15",
    urls = [
        "https://mirror.bazel.build/github.com/FFmpeg/FFmpeg/archive/n2.8.15.tar.gz",
        "https://github.com/FFmpeg/FFmpeg/archive/n2.8.15.tar.gz",
    ],
)

http_archive(
    name = "ffmpeg_3_4",
    build_file = "//third_party:ffmpeg_3_4.BUILD",
    sha256 = "bbccc87cd031498728bcc2dba5596a47e6fd92b2cec060a71feef65617a261fe",
    strip_prefix = "FFmpeg-n3.4.4",
    urls = [
        "https://mirror.bazel.build/github.com/FFmpeg/FFmpeg/archive/n3.4.4.tar.gz",
        "https://github.com/FFmpeg/FFmpeg/archive/n3.4.4.tar.gz",
    ],
)

http_archive(
    name = "libav_9_20",
    build_file = "//third_party:libav_9_20.BUILD",
    sha256 = "ecc2389bc857602450196c9240e1ebc59066980f5d42e977efe0f498145775d4",
    strip_prefix = "libav-9.20",
    urls = [
        "https://mirror.bazel.build/github.com/libav/libav/archive/v9.20.tar.gz",
        "https://github.com/libav/libav/archive/v9.20.tar.gz",
    ],
)

http_archive(
    name = "lmdb",
    build_file = "//third_party:lmdb.BUILD",
    sha256 = "f3927859882eb608868c8c31586bb7eb84562a40a6bf5cc3e13b6b564641ea28",
    strip_prefix = "lmdb-LMDB_0.9.22/libraries/liblmdb",
    urls = [
        "https://mirror.bazel.build/github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
        "https://github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
    ],
)

http_archive(
    name = "libtiff",
    build_file = "//third_party:libtiff.BUILD",
    sha256 = "2c52d11ccaf767457db0c46795d9c7d1a8d8f76f68b0b800a3dfe45786b996e4",
    strip_prefix = "tiff-4.0.10",
    urls = [
        "https://mirror.bazel.build/download.osgeo.org/libtiff/tiff-4.0.10.tar.gz",
        "https://download.osgeo.org/libtiff/tiff-4.0.10.tar.gz",
    ],
)

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "1d54cd95ed276c42c276e0a3df8ab33ee41968b73af14023c03a19db48f82e73",
    strip_prefix = "grpc-1.19.0",
    urls = [
        "https://mirror.bazel.build/github.com/grpc/grpc/archive/v1.19.0.tar.gz",
        "https://github.com/grpc/grpc/archive/v1.19.0.tar.gz",
    ],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

http_archive(
    name = "com_google_googleapis",
    patch_args = ["-p1"],
    patches = [
        "//third_party:googleapis.patch",
    ],
    sha256 = "ff903931e738b98418df9e95edeb00abe642007071dc3d579631d57957c3aa13",
    strip_prefix = "googleapis-c911062bb7a1c41a208957bed923b8750f3b6f28",
    urls = [
        "https://mirror.bazel.build/github.com/googleapis/googleapis/archive/c911062bb7a1c41a208957bed923b8750f3b6f28.tar.gz",
        "https://github.com/googleapis/googleapis/archive/c911062bb7a1c41a208957bed923b8750f3b6f28.tar.gz",
    ],
)

http_archive(
    name = "giflib",
    build_file = "//third_party:giflib.BUILD",
    sha256 = "34a7377ba834397db019e8eb122e551a49c98f49df75ec3fcc92b9a794a4f6d1",
    strip_prefix = "giflib-5.1.4",
    urls = [
        "https://mirror.bazel.build/ufpr.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
        "http://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
    ],
)

http_archive(
    name = "com_github_googlecloudplatform_google_cloud_cpp",
    sha256 = "06bc735a117ec7ea92ea580e7f2ffa4b1cd7539e0e04f847bf500588d7f0fe90",
    strip_prefix = "google-cloud-cpp-0.7.0",
    urls = [
        "https://mirror.bazel.build/github.com/googleapis/google-cloud-cpp/archive/v0.7.0.tar.gz",
        "https://github.com/googleapis/google-cloud-cpp/archive/v0.7.0.tar.gz",
    ],
)

http_archive(
    name = "com_github_googleapis_googleapis",
    build_file = "@com_github_googlecloudplatform_google_cloud_cpp//bazel:googleapis.BUILD",
    sha256 = "82ba91a41fb01305de4e8805c0a9270ed2035007161aa5a4ec60f887a499f5e9",
    strip_prefix = "googleapis-6a3277c0656219174ff7c345f31fb20a90b30b97",
    urls = [
        "https://github.com/google/googleapis/archive/6a3277c0656219174ff7c345f31fb20a90b30b97.zip",
    ],
)

http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "a31397714a353587413d307337d0b58f8a2e20e2b9d02f2e24e3463fa4eeda81",
    strip_prefix = "re2-2018-10-01",
    urls = [
        "https://mirror.bazel.build/github.com/google/re2/archive/2018-10-01.tar.gz",
        "https://github.com/google/re2/archive/2018-10-01.tar.gz",
    ],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
    strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
    urls = [
        "https://mirror.bazel.build/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
    ],
)

http_archive(
    name = "libarchive",
    build_file = "//third_party:libarchive.BUILD",
    sha256 = "720da414e7aebb255fcdaee106894e4d30e2472ac1390c2c15b70c84c7479658",
    strip_prefix = "libarchive-3.3.3",
    urls = [
        "https://mirror.bazel.build/github.com/libarchive/libarchive/archive/v3.3.3.tar.gz",
        "https://github.com/libarchive/libarchive/archive/v3.3.3.tar.gz",
    ],
)

http_archive(
    name = "libexpat",
    build_file = "//third_party:libexpat.BUILD",
    sha256 = "574499cba22a599393e28d99ecfa1e7fc85be7d6651d543045244d5b561cb7ff",
    strip_prefix = "libexpat-R_2_2_6/expat",
    urls = [
        "https://mirror.bazel.build/github.com/libexpat/libexpat/archive/R_2_2_6.tar.gz",
        "http://github.com/libexpat/libexpat/archive/R_2_2_6.tar.gz",
    ],
)

http_archive(
    name = "libapr1",
    build_file = "//third_party:libapr1.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:libapr1.patch",
    ],
    sha256 = "1a0909a1146a214a6ab9de28902045461901baab4e0ee43797539ec05b6dbae0",
    strip_prefix = "apr-1.6.5",
    urls = [
        "https://github.com/apache/apr/archive/1.6.5.tar.gz",
    ],
)

http_archive(
    name = "libaprutil1",
    build_file = "//third_party:libaprutil1.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:libaprutil1.patch",
    ],
    sha256 = "4c9ae319cedc16890fc2776920e7d529672dda9c3a9a9abd53bd80c2071b39af",
    strip_prefix = "apr-util-1.6.1",
    urls = [
        "https://github.com/apache/apr-util/archive/1.6.1.tar.gz",
    ],
)

http_archive(
    name = "mxml",
    build_file = "//third_party:mxml.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:mxml.patch",
    ],
    sha256 = "4d850d15cdd4fdb9e82817eb069050d7575059a9a2729c82b23440e4445da199",
    strip_prefix = "mxml-2.12",
    urls = [
        "https://github.com/michaelrsweet/mxml/archive/v2.12.tar.gz",
    ],
)

http_archive(
    name = "minini",
    build_file = "//third_party:minini.BUILD",
    sha256 = "a97dd5ac6811af95c8f2aeaa6894b3113377e78ffd585363c6848745760d0152",
    strip_prefix = "minIni-1.0",
    urls = [
        "https://github.com/ElasticDL/minIni/archive/v1.0.tar.gz",
    ],
)

http_archive(
    name = "aliyun_oss_c_sdk",
    build_file = "//third_party:oss_c_sdk.BUILD",
    sha256 = "6450d3970578c794b23e9e1645440c6f42f63be3f82383097660db5cf2fba685",
    strip_prefix = "aliyun-oss-c-sdk-3.7.0",
    urls = [
        "https://github.com/aliyun/aliyun-oss-c-sdk/archive/3.7.0.tar.gz",
    ],
)

# hdf5 header files are generated from:
#
# tar xzf hdf5-1.10.5.tar.gz
# cp -r hdf5-1.10.5 a
# cp -r hdf5-1.10.5 b
# docker run -i -t --rm -v $PWD/hdf5-1.10.5:/v -w /v --net=host ubuntu:14.04
# $ apt-get -y -qq update
# $ apt-get -y -qq install make gcc g++
# $ ./configure --enable-cxx
# $ make
# $ exit
# cp hdf5-1.10.5/src/H5pubconf.h b/src/H5pubconf.h
# cp hdf5-1.10.5/src/H5lib_settings.c b/src/H5lib_settings.c
# cp hdf5-1.10.5/src/H5Tinit.c b/src/H5Tinit.c
# diff -Naur a b > hdf5.patch
http_archive(
    name = "hdf5",
    build_file = "//third_party:hdf5.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:hdf5.patch",
    ],
    sha256 = "6d4ce8bf902a97b050f6f491f4268634e252a63dadd6656a1a9be5b7b7726fa8",
    strip_prefix = "hdf5-1.10.5",
    urls = [
        "https://mirror.bazel.build/support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.5.tar.gz",
        "https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.5.tar.gz",
    ],
)
