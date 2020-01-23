workspace(name = "org_tensorflow_io")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/toolchains/tf:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

load("//third_party/toolchains/gpu:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "local_config_cuda")

http_archive(
    name = "libwebp",
    build_file = "//third_party:libwebp.BUILD",
    sha256 = "424faab60a14cb92c2a062733b6977b4cc1e875a6398887c5911b3a1a6c56c51",
    strip_prefix = "libwebp-1.1.0",
    urls = [
        "https://github.com/webmproject/libwebp/archive/v1.1.0.tar.gz",
    ],
)

http_archive(
    name = "freetype",
    build_file = "//third_party:freetype.BUILD",
    sha256 = "3a60d391fd579440561bf0e7f31af2222bc610ad6ce4d9d7bd2165bca8669110",
    strip_prefix = "freetype-2.10.1",
    urls = [
        "https://download.savannah.gnu.org/releases/freetype/freetype-2.10.1.tar.gz",
    ],
)

http_archive(
    name = "easyexif",
    build_file = "//third_party:easyexif.BUILD",
    sha256 = "7a49a2617da70b318d1464625e1c5fd6d369d04aa1b23a270d3d0926d8669432",
    strip_prefix = "easyexif-19d15151c3f663813dc70cf9ff568d25ab6ff93b",
    urls = [
        "https://github.com/mayanklahiri/easyexif/archive/19d15151c3f663813dc70cf9ff568d25ab6ff93b.tar.gz",
    ],
)

http_archive(
    name = "stb",
    build_file = "//third_party:stb.BUILD",
    sha256 = "978de595fcc62448dbdc8ca8def7879fbe63245dd7f57c1898270e53a0abf95b",
    strip_prefix = "stb-052dce117ed989848a950308bd99eef55525dfb1",
    urls = [
        "https://github.com/nothings/stb/archive/052dce117ed989848a950308bd99eef55525dfb1.tar.gz",
    ],
)

http_archive(
    name = "boost",
    build_file = "//third_party:boost.BUILD",
    sha256 = "c66e88d5786f2ca4dbebb14e06b566fb642a1a6947ad8cc9091f9f445134143f",
    strip_prefix = "boost_1_72_0",
    urls = [
        "https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz",
        "https://downloads.sourceforge.net/project/boost/boost/1.72.0/boost_1_72_0.tar.gz",
    ],
)

http_archive(
    name = "avro",
    build_file = "//third_party:avro.BUILD",
    sha256 = "e382ac6685544ae9539084793ac0a4ffd377ba476ea756439625552e14d212b0",
    strip_prefix = "avro-release-1.9.1/lang/c++",
    urls = [
        "https://github.com/apache/avro/archive/release-1.9.1.tar.gz",
    ],
)

http_archive(
    name = "rapidjson",
    build_file = "//third_party:rapidjson.BUILD",
    sha256 = "30bd2c428216e50400d493b38ca33a25efb1dd65f79dfc614ab0c957a3ac2c28",
    strip_prefix = "rapidjson-418331e99f859f00bdc8306f69eba67e8693c55e",
    urls = [
        "https://github.com/miloyip/rapidjson/archive/418331e99f859f00bdc8306f69eba67e8693c55e.tar.gz",
    ],
)

http_archive(
    name = "lmdb",
    build_file = "//third_party:lmdb.BUILD",
    sha256 = "44602436c52c29d4f301f55f6fd8115f945469b868348e3cddaf91ab2473ea26",
    strip_prefix = "lmdb-LMDB_0.9.24/libraries/liblmdb",
    urls = [
        "https://github.com/LMDB/lmdb/archive/LMDB_0.9.24.tar.gz",
    ],
)

http_archive(
    name = "zlib",
    build_file = "//third_party:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
)

http_archive(
    name = "openexr",
    build_file = "//third_party:openexr.BUILD",
    sha256 = "4904c5ea7914a58f60a5e2fbc397be67e7a25c380d7d07c1c31a3eefff1c92f1",
    strip_prefix = "openexr-2.4.0",
    urls = [
        "https://github.com/openexr/openexr/archive/v2.4.0.tar.gz",
    ],
)

http_archive(
    name = "zstd",
    build_file = "//third_party:zstd.BUILD",
    sha256 = "a364f5162c7d1a455cc915e8e3cf5f4bd8b75d09bc0f53965b0c9ca1383c52c8",
    strip_prefix = "zstd-1.4.4",
    urls = [
        "https://github.com/facebook/zstd/archive/v1.4.4.tar.gz",
    ],
)

http_archive(
    name = "kafka",
    build_file = "//third_party:kafka.BUILD",
    sha256 = "465cab533ebc5b9ca8d97c90ab69e0093460665ebaf38623209cf343653c76d2",
    strip_prefix = "librdkafka-1.3.0",
    urls = [
        "https://github.com/edenhill/librdkafka/archive/v1.3.0.tar.gz",
    ],
)

http_archive(
    name = "com_google_absl",
    sha256 = "acd93f6baaedc4414ebd08b33bebca7c7a46888916101d8c0b8083573526d070",
    strip_prefix = "abseil-cpp-43ef2148c0936ebf7cb4be6b19927a9d9d145b8f",
    urls = [
        "http://mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz",
    ],
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
    sha256 = "69d9de9ec60a3080543b28a5334dbaf892ca34235b8bd8f8c1c01a33253926c1",
    strip_prefix = "arrow-apache-arrow-0.14.1",
    urls = [
        "https://mirror.bazel.build/github.com/apache/arrow/archive/apache-arrow-0.14.1.tar.gz",
        "https://github.com/apache/arrow/archive/apache-arrow-0.14.1.tar.gz",
    ],
)

http_archive(
    name = "thrift",
    build_file = "//third_party:thrift.BUILD",
    sha256 = "b7452d1873c6c43a580d2b4ae38cfaf8fa098ee6dc2925bae98dce0c010b1366",
    strip_prefix = "thrift-0.12.0",
    urls = [
        "https://mirror.bazel.build/github.com/apache/thrift/archive/0.12.0.tar.gz",
        "https://github.com/apache/thrift/archive/0.12.0.tar.gz",
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
    name = "libtiff",
    build_file = "//third_party:libtiff.BUILD",
    sha256 = "5d29f32517dadb6dbcd1255ea5bbc93a2b54b94fbf83653b4d65c7d6775b8634",
    strip_prefix = "tiff-4.1.0",
    urls = [
        "https://download.osgeo.org/libtiff/tiff-4.1.0.tar.gz",
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
    name = "rules_python",
    sha256 = "c911dc70f62f507f3a361cbc21d6e0d502b91254382255309bc60b7a0f48de28",
    strip_prefix = "rules_python-38f86fb55b698c51e8510c807489c9f4e047480e",
    urls = [
        "https://github.com/bazelbuild/rules_python/archive/38f86fb55b698c51e8510c807489c9f4e047480e.tar.gz",
    ],
)

load("@rules_python//python:pip.bzl", "pip3_import")

pip3_import(
    name = "lint_dependencies",
    requirements = "//tools/lint:requirements.txt",
)

load("@lint_dependencies//:requirements.bzl", "pip_install")

pip_install()

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "2fcb7f1ab160d6fd3aaade64520be3e5446fc4c6fa7ba6581afdc4e26094bd81",
    strip_prefix = "grpc-1.26.0",
    urls = [
        "https://github.com/grpc/grpc/archive/v1.26.0.tar.gz",
    ],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@rules_python//python:pip.bzl", "pip_repositories")

pip3_import(
    name = "grpc_python_dependencies",
    requirements = "@com_github_grpc_grpc//:requirements.bazel.txt",
)

load("@grpc_python_dependencies//:requirements.bzl", "pip_install")

pip_repositories()

pip_install()

# TODO(https://github.com/grpc/grpc/issues/19835): Remove.
load("@upb//bazel:workspace_deps.bzl", "upb_deps")

upb_deps()

load("@build_bazel_rules_apple//apple:repositories.bzl", "apple_rules_dependencies")

apple_rules_dependencies()

load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")

apple_support_dependencies()

http_archive(
    name = "com_github_googleapis_google_cloud_cpp",
    sha256 = "35058ff14e4f9f49f78da2f1bbf1c03f27e8e40ec65c51f62720346e99803392",
    strip_prefix = "google-cloud-cpp-0.13.0",
    urls = [
        "https://mirror.bazel.build/github.com/googleapis/google-cloud-cpp/archive/v0.13.0.tar.gz",
        "https://github.com/googleapis/google-cloud-cpp/archive/v0.13.0.tar.gz",
    ],
)

http_archive(
    name = "com_google_googleapis",
    build_file = "@com_github_googleapis_google_cloud_cpp//bazel:googleapis.BUILD",
    sha256 = "cb531e445115e28054a33ad968c2d7d8ade4693721866ce1b9adf9a78762c032",
    strip_prefix = "googleapis-960b76b1f0c46d12610088977d1129cc7405f3dc",
    urls = [
        "https://github.com/googleapis/googleapis/archive/960b76b1f0c46d12610088977d1129cc7405f3dc.tar.gz",
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
    name = "jsoncpp_git",
    build_file = "//third_party:jsoncpp.BUILD",
    sha256 = "c49deac9e0933bcb7044f08516861a2d560988540b23de2ac1ad443b219afdb6",
    strip_prefix = "jsoncpp-1.8.4",
    urls = [
        "http://mirror.tensorflow.org/github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
        "https://github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
    ],
)

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "ae8c36ff6e565f674c7a3692d6a9ea1096e4c1ade497272c2108a810fb39acd2",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/rules_go/releases/download/0.19.4/rules_go-0.19.4.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/0.19.4/rules_go-0.19.4.tar.gz",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "7fc87f4170011201b1690326e8c16c5d802836e3a0d617d8f75c3af2b23180c4",
    urls = ["https://github.com/bazelbuild/bazel-gazelle/releases/download/0.18.2/bazel-gazelle-0.18.2.tar.gz"],
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

go_repository(
    name = "com_github_matttproud_golang_protobuf_extensionsn",
    commit = "c182affec369e30f25d3eb8cd8a478dee585ae7d",
    importpath = "github.com/matttproud/golang_protobuf_extensions",
)

go_repository(
    name = "com_github_prometheus_client_model",
    commit = "14fe0d1b01d4d5fc031dd4bec1823bd3ebbe8016",
    importpath = "github.com/prometheus/client_model",
)

go_repository(
    name = "com_github_prometheus_prom2json",
    build_extra_args = ["-exclude=vendor"],
    importpath = "github.com/prometheus/prom2json",
    tag = "v1.2.2",
)

http_archive(
    name = "dcmtk",
    build_file = "//third_party:dcmtk.BUILD",
    sha256 = "a93ff354fae091689a0740a1000cde7d4378fdf733aef9287a70d7091efa42c0",
    strip_prefix = "dcmtk-3.6.4",
    urls = [
        "https://dicom.offis.de/download/dcmtk/dcmtk364/dcmtk-3.6.4.tar.gz",
    ],
)

http_archive(
    name = "nucleus",
    build_file = "//third_party:nucleus.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:nucleus.patch",
    ],
    sha256 = "aa865d3509ba8f3527392303bd95a11f48f19e68197b3d1d0bae9fab004bee87",
    strip_prefix = "nucleus-0.4.1",
    urls = [
        "https://github.com/google/nucleus/archive/0.4.1.tar.gz",
    ],
)

http_archive(
    name = "bzip2",
    build_file = "//third_party:bzip2.BUILD",
    sha256 = "ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269",
    strip_prefix = "bzip2-1.0.8",
    urls = [
        "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz",
    ],
)

http_archive(
    name = "com_googlesource_code_cctz",
    strip_prefix = "cctz-master",
    urls = ["https://github.com/google/cctz/archive/master.zip"],
)

# This is the 1.9 release of htslib.
http_archive(
    name = "htslib",
    build_file = "//third_party:htslib.BUILD",
    sha256 = "c4d3ae84014f8a80f5011521f391e917bc3b4f6ebd78e97f238472e95849ec14",
    strip_prefix = "htslib-1.9",
    urls = [
        "https://github.com/samtools/htslib/archive/1.9.zip",
    ],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "43c9b882fa921923bcba764453f4058d102bece35a37c9f6383c713004aacff1",
    strip_prefix = "rules_closure-9889e2348259a5aad7e805547c1a0cf311cfcd91",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",  # 2018-12-21
    ],
)

# bazel_skylib is now a required dependency of protobuf_archive.
http_archive(
    name = "bazel_skylib",
    sha256 = "bbccf674aa441c266df9894182d80de104cabd19be98be002f6d478aaa31574d",
    strip_prefix = "bazel-skylib-2169ae1c374aab4a09aa90e65efe1a3aad4e279b",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/2169ae1c374aab4a09aa90e65efe1a3aad4e279b.tar.gz"],
)

http_archive(
    name = "double_conversion",
    build_file = "//third_party:double_conversion.BUILD",
    sha256 = "2f7fbffac0d98d201ad0586f686034371a6d152ca67508ab611adc2386ad30de",
    strip_prefix = "double-conversion-3992066a95b823efc8ccc1baf82a1cfc73f6e9b8",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
        "https://github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
    ],
)

http_archive(
    name = "xz",
    build_file = "//third_party:xz.BUILD",
    sha256 = "b512f3b726d3b37b6dc4c8570e137b9311e7552e8ccbab4d39d47ce5f4177145",
    strip_prefix = "xz-5.2.4",
    urls = [
        "https://tukaani.org/xz/xz-5.2.4.tar.gz",
    ],
)

http_archive(
    name = "com_grail_bazel_toolchain",
    strip_prefix = "bazel-toolchain-0.4.4",
    urls = ["https://github.com/grailbio/bazel-toolchain/archive/0.4.4.tar.gz"],
)

load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "8.0.0",
)

http_archive(
    name = "vorbis",
    build_file = "//third_party:vorbis.BUILD",
    sha256 = "43fc4bc34f13da15b8acfa72fd594678e214d1cab35fc51d3a54969a725464eb",
    strip_prefix = "vorbis-1.3.6",
    urls = [
        "https://github.com/xiph/vorbis/archive/v1.3.6.tar.gz",
    ],
)

http_archive(
    name = "ogg",
    build_file = "//third_party:ogg.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:ogg.patch",
    ],
    sha256 = "3da31a4eb31534b6f878914b7379b873c280e610649fe5c07935b3d137a828bc",
    strip_prefix = "ogg-1.3.4",
    urls = [
        "https://github.com/xiph/ogg/archive/v1.3.4.tar.gz",
    ],
)

http_archive(
    name = "flac",
    build_file = "//third_party:flac.BUILD",
    sha256 = "668cdeab898a7dd43cf84739f7e1f3ed6b35ece2ef9968a5c7079fe9adfe1689",
    strip_prefix = "flac-1.3.3",
    urls = [
        "https://github.com/xiph/flac/archive/1.3.3.tar.gz",
    ],
)
