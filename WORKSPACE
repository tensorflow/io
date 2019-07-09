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
        "https://downloads.sourceforge.net/project/boost/boost/1.67.0/boost_1_67_0.tar.gz",
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
    name = "com_github_azure_azure_storage_cpplite",
    build_file = "//third_party:azure.BUILD",
    sha256 = "a0c315120ba15c4fae64aacecc7473f6a6b2be765d493ec5d183d774eefc10eb",
    strip_prefix = "azure-storage-cpplite-d57610340eae795d57959db106fd7216426d63b7",
    urls = [
        "https://github.com/Azure/azure-storage-cpplite/archive/d57610340eae795d57959db106fd7216426d63b7.zip",
        "https://mirror.bazel.build/github.com/Azure/azure-storage-cpplite/archive/d57610340eae795d57959db106fd7216426d63b7.zip",
    ],
)

http_archive(
    name = "util_linux",
    build_file = "//third_party:uuid.BUILD",
    sha256 = "2483d5a42bc39575fc215c6994554f5169db777262d606ebe9cd8d5f37557f72",
    strip_prefix = "util-linux-2.32.1",
    urls = [
        "https://github.com/karelzak/util-linux/archive/v2.32.1.tar.gz",
        "https://mirror.bazel.build/github.com/karelzak/util-linux/archive/v2.32.1.tar.gz",
    ],
)

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "1bf082fb3016154d3f806da8eb5876caf05743da4b2e8130fadd000df74b5bb6",
    strip_prefix = "grpc-1.21.1",
    urls = [
        "https://mirror.bazel.build/github.com/grpc/grpc/archive/v1.21.1.tar.gz",
        "https://github.com/grpc/grpc/archive/v1.21.1.tar.gz",
    ],
)

# 3.7.1 with a fix to BUILD file
http_archive(
    name = "com_google_protobuf",
    sha256 = "1c020fafc84acd235ec81c6aac22d73f23e85a700871466052ff231d69c1b17a",
    strip_prefix = "protobuf-5902e759108d14ee8e6b0b07653dac2f4e70ac73",
    urls = [
        "http://mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/5902e759108d14ee8e6b0b07653dac2f4e70ac73.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/5902e759108d14ee8e6b0b07653dac2f4e70ac73.tar.gz",
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

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
    name = "com_github_googleapis_google_cloud_cpp",
    sha256 = "3abe2cf553ce33ff58d23848ae716cd2fcabfd454b89f6f65a92ed261244c1df",
    strip_prefix = "google-cloud-cpp-0.11.0",
    urls = [
        "https://mirror.bazel.build/github.com/googleapis/google-cloud-cpp/archive/v0.11.0.tar.gz",
        "https://github.com/googleapis/google-cloud-cpp/archive/v0.11.0.tar.gz",
    ],
)

# Manually load com_google_googleapis as we need a patch for pubsub
# The patch file was generated from:
# diff -Naur a b > third_party/googleapis.patch
http_archive(
    name = "com_google_googleapis",
    build_file = "@com_github_googleapis_google_cloud_cpp//bazel:googleapis.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:googleapis.patch",
    ],
    sha256 = "90bcdf27b41b1c3900838fe4edaf89080ca67026608817946b4ae5e2b925c711",
    strip_prefix = "googleapis-7152063cb170d23c5c110e243711d8eb6fda6a1c",
    urls = [
        "https://github.com/googleapis/googleapis/archive/7152063cb170d23c5c110e243711d8eb6fda6a1c.tar.gz",
    ],
)

load("@com_github_googleapis_google_cloud_cpp//bazel:google_cloud_cpp_deps.bzl", "google_cloud_cpp_deps")

google_cloud_cpp_deps()

load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")

# Configure @com_google_googleapis to only compile C++ and gRPC:
switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,  # C++ support is only "Partially implemented", roll our own.
    grpc = True,
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
    sha256 = "9bf1fe5182a604b4135edc1a425ae356c9ad15e9b23f9f12a02e80184c3a249c",
    strip_prefix = "googletest-release-1.8.1",
    urls = [
        "https://mirror.bazel.build/github.com/google/googletest/archive/release-1.8.1.tar.gz",
        "https://github.com/google/googletest/archive/release-1.8.1.tar.gz",
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
# $ apt-get -y -qq install make gcc g++ libz-dev
# $ ./configure --enable-cxx --with-zlib
# $ make
# $ exit
# mkdir -p b/linux/src
# cp hdf5-1.10.5/src/H5pubconf.h b/linux/src/H5pubconf.h
# cp hdf5-1.10.5/src/H5lib_settings.c b/linux/src/H5lib_settings.c
# cp hdf5-1.10.5/src/H5Tinit.c b/linux/src/H5Tinit.c
# diff -Naur a b > hdf5.linux.patch
#
# On darwin, change to:
#
# mkdir -p b/darwin/src
# cp hdf5-1.10.5/src/H5pubconf.h b/darwin/src/H5pubconf.h
# cp hdf5-1.10.5/src/H5lib_settings.c b/darwin/src/H5lib_settings.c
# cp hdf5-1.10.5/src/H5Tinit.c b/darwin/src/H5Tinit.c
# diff -Naur a b > hdf5.darwin.patch
http_archive(
    name = "hdf5",
    build_file = "//third_party:hdf5.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:hdf5.linux.patch",
        "//third_party:hdf5.darwin.patch",
    ],
    sha256 = "6d4ce8bf902a97b050f6f491f4268634e252a63dadd6656a1a9be5b7b7726fa8",
    strip_prefix = "hdf5-1.10.5",
    urls = [
        "https://mirror.bazel.build/support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.5.tar.gz",
        "https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.5.tar.gz",
    ],
)

http_archive(
    name = "avro",
    build_file = "//third_party:avro.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:avro.patch",
    ],
    sha256 = "42fbe407263ec174d448121cd0e20adcd75c43cb9f192b97476d9b99fa69faf3",
    strip_prefix = "avro-release-1.9.0-rc4/lang/c++",
    urls = [
        "https://mirror.bazel.build/github.com/apache/avro/archive/release-1.9.0-rc4.tar.gz",
        "https://github.com/apache/avro/archive/release-1.9.0-rc4.tar.gz",
    ],
)

http_archive(
    name = "freetype",
    build_file = "//third_party:freetype.BUILD",
    sha256 = "955e17244e9b38adb0c98df66abb50467312e6bb70eac07e49ce6bd1a20e809a",
    strip_prefix = "freetype-2.10.0",
    urls = [
        "https://mirror.bazel.build/download.savannah.gnu.org/releases/freetype/freetype-2.10.0.tar.gz",
        "https://download.savannah.gnu.org/releases/freetype/freetype-2.10.0.tar.gz",
    ],
)

http_archive(
    name = "jsoncpp_git",
    build_file = "//third_party:jsoncpp.BUILD",
    sha256 = "c49deac9e0933bcb7044f08516861a2d560988540b23de2ac1ad443b219afdb6",
    strip_prefix = "jsoncpp-1.8.4",
    urls = [
        "http://mirror.tensorflow.org/github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
        "https://github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
    ],
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Note: patch is needed as we need to resolve multiple zlib dependencies.
# Patch was created with:
# diff -Naur a b > rules_go.patch
http_archive(
    name = "io_bazel_rules_go",
    patch_args = ["-p1"],
    patches = [
        "//third_party:rules_go.patch",
    ],
    sha256 = "f04d2373bcaf8aa09bccb08a98a57e721306c8f6043a2a0ee610fd6853dcde3d",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/rules_go/releases/download/0.18.6/rules_go-0.18.6.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/0.18.6/rules_go-0.18.6.tar.gz",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "3c681998538231a2d24d0c07ed5a7658cb72bfb5fd4bf9911157c0e9ac6a2687",
    urls = ["https://github.com/bazelbuild/bazel-gazelle/releases/download/0.17.0/bazel-gazelle-0.17.0.tar.gz"],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains()

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies", "go_repository")

gazelle_dependencies()

go_repository(
    name = "com_github_prometheus_common",
    importpath = "github.com/prometheus/common",
    tag = "v0.4.1",
)

go_repository(
    name = "com_github_prometheus_client_golang",
    importpath = "github.com/prometheus/client_golang",
    tag = "v0.9.3",
)
