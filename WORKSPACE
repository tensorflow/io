workspace(name = "org_tensorflow_io")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("//third_party/toolchains/tf:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

load("//third_party/toolchains/gpu:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "local_config_cuda")

http_archive(
    name = "com_google_protobuf",
    sha256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",
    strip_prefix = "protobuf-3.9.2",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
        "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
    ],
)

http_archive(
    name = "libwebp",
    build_file = "//third_party:libwebp.BUILD",
    sha256 = "424faab60a14cb92c2a062733b6977b4cc1e875a6398887c5911b3a1a6c56c51",
    strip_prefix = "libwebp-1.1.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/webmproject/libwebp/archive/v1.1.0.tar.gz",
        "https://github.com/webmproject/libwebp/archive/v1.1.0.tar.gz",
    ],
)

http_archive(
    name = "freetype",
    build_file = "//third_party:freetype.BUILD",
    sha256 = "3a60d391fd579440561bf0e7f31af2222bc610ad6ce4d9d7bd2165bca8669110",
    strip_prefix = "freetype-2.10.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/download.savannah.gnu.org/releases/freetype/freetype-2.10.1.tar.gz",
        "https://download.savannah.gnu.org/releases/freetype/freetype-2.10.1.tar.gz",
    ],
)

http_archive(
    name = "easyexif",
    build_file = "//third_party:easyexif.BUILD",
    sha256 = "7a49a2617da70b318d1464625e1c5fd6d369d04aa1b23a270d3d0926d8669432",
    strip_prefix = "easyexif-19d15151c3f663813dc70cf9ff568d25ab6ff93b",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/mayanklahiri/easyexif/archive/19d15151c3f663813dc70cf9ff568d25ab6ff93b.tar.gz",
        "https://github.com/mayanklahiri/easyexif/archive/19d15151c3f663813dc70cf9ff568d25ab6ff93b.tar.gz",
    ],
)

http_archive(
    name = "stb",
    build_file = "//third_party:stb.BUILD",
    sha256 = "978de595fcc62448dbdc8ca8def7879fbe63245dd7f57c1898270e53a0abf95b",
    strip_prefix = "stb-052dce117ed989848a950308bd99eef55525dfb1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nothings/stb/archive/052dce117ed989848a950308bd99eef55525dfb1.tar.gz",
        "https://github.com/nothings/stb/archive/052dce117ed989848a950308bd99eef55525dfb1.tar.gz",
    ],
)

http_archive(
    name = "boost",
    build_file = "//third_party:boost.BUILD",
    sha256 = "c66e88d5786f2ca4dbebb14e06b566fb642a1a6947ad8cc9091f9f445134143f",
    strip_prefix = "boost_1_72_0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz",
        "https://storage.googleapis.com/mirror.tensorflow.org/downloads.sourceforge.net/project/boost/boost/1.72.0/boost_1_72_0.tar.gz",
        "https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz",
        "https://downloads.sourceforge.net/project/boost/boost/1.72.0/boost_1_72_0.tar.gz",
    ],
)

http_archive(
    name = "avro",
    build_file = "//third_party:avro.BUILD",
    sha256 = "8fd1f850ce37e60835e6d8335c0027a959aaa316773da8a9660f7d33a66ac142",
    strip_prefix = "avro-release-1.10.1/lang/c++",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/avro/archive/release-1.10.1.tar.gz",
        "https://github.com/apache/avro/archive/release-1.10.1.tar.gz",
    ],
)

http_archive(
    name = "rapidjson",
    build_file = "//third_party:rapidjson.BUILD",
    sha256 = "30bd2c428216e50400d493b38ca33a25efb1dd65f79dfc614ab0c957a3ac2c28",
    strip_prefix = "rapidjson-418331e99f859f00bdc8306f69eba67e8693c55e",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/miloyip/rapidjson/archive/418331e99f859f00bdc8306f69eba67e8693c55e.tar.gz",
        "https://github.com/miloyip/rapidjson/archive/418331e99f859f00bdc8306f69eba67e8693c55e.tar.gz",
    ],
)

http_archive(
    name = "lmdb",
    build_file = "//third_party:lmdb.BUILD",
    sha256 = "44602436c52c29d4f301f55f6fd8115f945469b868348e3cddaf91ab2473ea26",
    strip_prefix = "lmdb-LMDB_0.9.24/libraries/liblmdb",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/LMDB/lmdb/archive/LMDB_0.9.24.tar.gz",
        "https://github.com/LMDB/lmdb/archive/LMDB_0.9.24.tar.gz",
    ],
)

http_archive(
    name = "zlib",
    build_file = "//third_party:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
)

http_archive(
    name = "openexr",
    build_file = "//third_party:openexr.BUILD",
    sha256 = "4904c5ea7914a58f60a5e2fbc397be67e7a25c380d7d07c1c31a3eefff1c92f1",
    strip_prefix = "openexr-2.4.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/openexr/openexr/archive/v2.4.0.tar.gz",
        "https://github.com/openexr/openexr/archive/v2.4.0.tar.gz",
    ],
)

new_git_repository(
    name = "libyuv",
    build_file = "//third_party:libyuv.BUILD",
    commit = "7f00d67d7c279f13b73d3be9c2d85873a7e2fbaf",
    remote = "https://chromium.googlesource.com/libyuv/libyuv",
)

http_archive(
    name = "libgeotiff",
    build_file = "//third_party:libgeotiff.BUILD",
    sha256 = "9452dadd126223a22ce6b97d202066d3873792aaefa7ce739519635a3fe34034",
    strip_prefix = "libgeotiff-1.6.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/OSGeo/libgeotiff/releases/download/1.6.0/libgeotiff-1.6.0.zip",
        "https://github.com/OSGeo/libgeotiff/releases/download/1.6.0/libgeotiff-1.6.0.zip",
    ],
)

http_archive(
    name = "proj",
    build_file = "//third_party:proj.BUILD",
    sha256 = "219c6e11b2baa9a3e2bd7ec54ce19702909591032cf6f7d1004b406f10b7c9ad",
    strip_prefix = "proj-7.2.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/OSGeo/PROJ/releases/download/7.2.1/proj-7.2.1.zip",
        "https://github.com/OSGeo/PROJ/releases/download/7.2.1/proj-7.2.1.zip",
    ],
)

http_archive(
    name = "sqlite",
    build_file = "//third_party:sqlite.BUILD",
    sha256 = "adf051d4c10781ea5cfabbbc4a2577b6ceca68590d23b58b8260a8e24cc5f081",
    strip_prefix = "sqlite-amalgamation-3300100",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/www.sqlite.org/2019/sqlite-amalgamation-3300100.zip",
        "https://www.sqlite.org/2019/sqlite-amalgamation-3300100.zip",
    ],
)

http_archive(
    name = "com_github_azure_azure_storage_cpplite",
    build_file = "//third_party:azure.BUILD",
    patch_cmds = [
        "sed -i.bak 's/struct stat/struct_stat/' src/blob/blob_client_wrapper.cpp",
        "echo '' >> include/base64.h",
        "echo '#include <stdexcept>' >> include/base64.h",
    ],
    sha256 = "25f34354fb0400ffe1b5a5c09c793c9fc8104d375910f6c84ab10fa50c0059cb",
    strip_prefix = "azure-storage-cpplite-0.3.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Azure/azure-storage-cpplite/archive/v0.3.0.tar.gz",
        "https://github.com/Azure/azure-storage-cpplite/archive/v0.3.0.tar.gz",
    ],
)

http_archive(
    name = "hdf5",
    build_file = "//third_party:hdf5.BUILD",
    sha256 = "5f9a3ee85db4ea1d3b1fa9159352aebc2af72732fc2f58c96a3f0768dba0e9aa",
    strip_prefix = "hdf5-1.10.6",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz",
        "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz",
    ],
)

http_archive(
    name = "zstd",
    build_file = "//third_party:zstd.BUILD",
    sha256 = "a364f5162c7d1a455cc915e8e3cf5f4bd8b75d09bc0f53965b0c9ca1383c52c8",
    strip_prefix = "zstd-1.4.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/facebook/zstd/archive/v1.4.4.tar.gz",
        "https://github.com/facebook/zstd/archive/v1.4.4.tar.gz",
    ],
)

http_archive(
    name = "lz4",
    build_file = "//third_party:lz4.BUILD",
    sha256 = "658ba6191fa44c92280d4aa2c271b0f4fbc0e34d249578dd05e50e76d0e5efcc",
    strip_prefix = "lz4-1.9.2",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/lz4/lz4/archive/v1.9.2.tar.gz",
        "https://github.com/lz4/lz4/archive/v1.9.2.tar.gz",
    ],
)

http_archive(
    name = "brotli",
    build_file = "//third_party:brotli.BUILD",
    sha256 = "4c61bfb0faca87219ea587326c467b95acb25555b53d1a421ffa3c8a9296ee2c",
    strip_prefix = "brotli-1.0.7",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/brotli/archive/v1.0.7.tar.gz",
        "https://github.com/google/brotli/archive/v1.0.7.tar.gz",
    ],
)

http_archive(
    name = "snappy",
    build_file = "//third_party:snappy.BUILD",
    sha256 = "16b677f07832a612b0836178db7f374e414f94657c138e6993cbfc5dcc58651f",
    strip_prefix = "snappy-1.1.8",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/snappy/archive/1.1.8.tar.gz",
        "https://github.com/google/snappy/archive/1.1.8.tar.gz",
    ],
)

http_archive(
    name = "double-conversion",
    sha256 = "a63ecb93182134ba4293fd5f22d6e08ca417caafa244afaa751cbfddf6415b13",
    strip_prefix = "double-conversion-3.1.5",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/double-conversion/archive/v3.1.5.tar.gz",
        "https://github.com/google/double-conversion/archive/v3.1.5.tar.gz",
    ],
)

http_archive(
    name = "thrift",
    build_file = "//third_party:thrift.BUILD",
    sha256 = "5da60088e60984f4f0801deeea628d193c33cec621e78c8a43a5d8c4055f7ad9",
    strip_prefix = "thrift-0.13.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/thrift/archive/v0.13.0.tar.gz",
        "https://github.com/apache/thrift/archive/v0.13.0.tar.gz",
    ],
)

http_archive(
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    patch_cmds = [
        # TODO: Remove the fowllowing once arrow issue is resolved.
        """sed -i.bak 's/type_traits/std::max<int16_t>(sizeof(int16_t), type_traits/g' cpp/src/parquet/column_reader.cc""",
        """sed -i.bak 's/value_byte_size/value_byte_size)/g' cpp/src/parquet/column_reader.cc""",
    ],
    sha256 = "a27971e2a71c412ae43d998b7b6d06201c7a3da382c804dcdc4a8126ccbabe67",
    strip_prefix = "arrow-apache-arrow-4.0.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/arrow/archive/apache-arrow-4.0.0.tar.gz",
        "https://github.com/apache/arrow/archive/apache-arrow-4.0.0.tar.gz",
    ],
)

http_archive(
    name = "kafka",
    build_file = "//third_party:kafka.BUILD",
    patch_cmds = [
        "rm -f src/win32_config.h",
        # TODO: Remove the fowllowing once librdkafka issue is resolved.
        """sed -i.bak '\\|rd_kafka_log(rk,|,/ exceeded);/ s/^/\\/\\//' src/rdkafka_cgrp.c""",
    ],
    sha256 = "f7fee59fdbf1286ec23ef0b35b2dfb41031c8727c90ced6435b8cf576f23a656",
    strip_prefix = "librdkafka-1.5.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/edenhill/librdkafka/archive/v1.5.0.tar.gz",
        "https://github.com/edenhill/librdkafka/archive/v1.5.0.tar.gz",
    ],
)

http_archive(
    name = "dcmtk",
    build_file = "//third_party:dcmtk.BUILD",
    sha256 = "a05178665f21896dbb0974106dba1ad144975414abd760b4cf8f5cc979f9beb9",
    strip_prefix = "dcmtk-3.6.5",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/dicom.offis.de/download/dcmtk/dcmtk365/dcmtk-3.6.5.tar.gz",
        "https://dicom.offis.de/download/dcmtk/dcmtk365/dcmtk-3.6.5.tar.gz",
    ],
)

http_archive(
    name = "openjpeg",
    build_file = "//third_party:openjpeg.BUILD",
    sha256 = "8702ba68b442657f11aaeb2b338443ca8d5fb95b0d845757968a7be31ef7f16d",
    strip_prefix = "openjpeg-2.4.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/uclouvain/openjpeg/archive/v2.4.0.tar.gz",
        "https://github.com/uclouvain/openjpeg/archive/v2.4.0.tar.gz",
    ],
)

http_archive(
    name = "libtiff",
    build_file = "//third_party:libtiff.BUILD",
    sha256 = "eb0484e568ead8fa23b513e9b0041df7e327f4ee2d22db5a533929dfc19633cb",
    strip_prefix = "tiff-4.2.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/download.osgeo.org/libtiff/tiff-4.2.0.tar.gz",
        "https://download.osgeo.org/libtiff/tiff-4.2.0.tar.gz",
    ],
)

http_archive(
    name = "libpng",
    build_file = "//third_party:libpng.BUILD",
    sha256 = "ca74a0dace179a8422187671aee97dd3892b53e168627145271cad5b5ac81307",
    strip_prefix = "libpng-1.6.37",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/glennrp/libpng/archive/v1.6.37.tar.gz",
        "https://github.com/glennrp/libpng/archive/v1.6.37.tar.gz",
    ],
)

http_archive(
    name = "fmjpeg2koj",
    build_file = "//third_party:fmjpeg2koj.BUILD",
    sha256 = "a8563307cb09161633479aff0880368ed57396f6d532facba973cf303d699717",
    strip_prefix = "fmjpeg2koj-6de80e15a43a4d1c411109aea388007afee24263",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/DraconPern/fmjpeg2koj/archive/6de80e15a43a4d1c411109aea388007afee24263.tar.gz",
        "https://github.com/DraconPern/fmjpeg2koj/archive/6de80e15a43a4d1c411109aea388007afee24263.tar.gz",
    ],
)

http_archive(
    name = "aws-checksums",
    build_file = "//third_party:aws-checksums.BUILD",
    sha256 = "6e6bed6f75cf54006b6bafb01b3b96df19605572131a2260fddaf0e87949ced0",
    strip_prefix = "aws-checksums-0.1.5",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
        "https://github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
    ],
)

http_archive(
    name = "aws-c-common",
    build_file = "//third_party:aws-c-common.BUILD",
    sha256 = "01c2a58553a37b3aa5914d9e0bf7bf14507ff4937bc5872a678892ca20fcae1f",
    strip_prefix = "aws-c-common-0.4.29",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-c-common/archive/v0.4.29.tar.gz",
        "https://github.com/awslabs/aws-c-common/archive/v0.4.29.tar.gz",
    ],
)

http_archive(
    name = "aws-c-event-stream",
    build_file = "//third_party:aws-c-event-stream.BUILD",
    sha256 = "31d880d1c868d3f3df1e1f4b45e56ac73724a4dc3449d04d47fc0746f6f077b6",
    strip_prefix = "aws-c-event-stream-0.1.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
        "https://github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
    ],
)

http_archive(
    name = "aws-sdk-cpp",
    build_file = "//third_party:aws-sdk-cpp.BUILD",
    patch_cmds = [
        """sed -i.bak 's/UUID::RandomUUID/Aws::Utils::UUID::RandomUUID/g' aws-cpp-sdk-core/source/client/AWSClient.cpp""",
    ],
    sha256 = "758174f9788fed6cc1e266bcecb20bf738bd5ef1c3d646131c9ed15c2d6c5720",
    strip_prefix = "aws-sdk-cpp-1.7.336",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/aws/aws-sdk-cpp/archive/1.7.336.tar.gz",
        "https://github.com/aws/aws-sdk-cpp/archive/1.7.336.tar.gz",
    ],
)

http_archive(
    name = "com_google_absl",
    sha256 = "62c27e7a633e965a2f40ff16b487c3b778eae440bab64cad83b34ef1cbe3aa93",
    strip_prefix = "abseil-cpp-6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c.tar.gz",
    ],
)

http_archive(
    name = "boringssl",
    patch_cmds = [
        """sed -i.bak 's/bio.c",/bio.c","src\\/decrepit\\/bio\\/base64_bio.c",/g' BUILD.generated.bzl""",
    ],
    sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
    strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
        "https://github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
    ],
)

http_archive(
    name = "curl",
    build_file = "//third_party:curl.BUILD",
    sha256 = "01ae0c123dee45b01bbaef94c0bc00ed2aec89cb2ee0fd598e0d302a6b5e0a98",
    strip_prefix = "curl-7.69.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/curl.haxx.se/download/curl-7.69.1.tar.gz",
        "https://curl.haxx.se/download/curl-7.69.1.tar.gz",
    ],
)

http_archive(
    name = "com_github_google_flatbuffers",
    sha256 = "62f2223fb9181d1d6338451375628975775f7522185266cd5296571ac152bc45",
    strip_prefix = "flatbuffers-1.12.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v1.12.0.tar.gz",
        "https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz",
    ],
)

http_archive(
    name = "xz",
    build_file = "//third_party:xz.BUILD",
    sha256 = "0d2b89629f13dd1a0602810529327195eff5f62a0142ccd65b903bc16a4ac78a",
    strip_prefix = "xz-5.2.5",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/xz-mirror/xz/archive/v5.2.5.tar.gz",
        "https://github.com/xz-mirror/xz/archive/v5.2.5.tar.gz",
    ],
)

http_archive(
    name = "util_linux",
    build_file = "//third_party:uuid.BUILD",
    sha256 = "2483d5a42bc39575fc215c6994554f5169db777262d606ebe9cd8d5f37557f72",
    strip_prefix = "util-linux-2.32.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/karelzak/util-linux/archive/v2.32.1.tar.gz",
        "https://github.com/karelzak/util-linux/archive/v2.32.1.tar.gz",
    ],
)

http_archive(
    name = "ffmpeg_2_8",
    build_file = "//third_party:ffmpeg_2_8.BUILD",
    sha256 = "8ba1b91a14431fe37091936c3a34469d7473965ab9edde0343c88f2d920bd918",
    strip_prefix = "FFmpeg-n2.8.15",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/FFmpeg/FFmpeg/archive/n2.8.15.tar.gz",
        "https://github.com/FFmpeg/FFmpeg/archive/n2.8.15.tar.gz",
    ],
)

http_archive(
    name = "ffmpeg_3_4",
    build_file = "//third_party:ffmpeg_3_4.BUILD",
    sha256 = "bbccc87cd031498728bcc2dba5596a47e6fd92b2cec060a71feef65617a261fe",
    strip_prefix = "FFmpeg-n3.4.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/FFmpeg/FFmpeg/archive/n3.4.4.tar.gz",
        "https://github.com/FFmpeg/FFmpeg/archive/n3.4.4.tar.gz",
    ],
)

http_archive(
    name = "ffmpeg_4_2",
    build_file = "//third_party:ffmpeg_4_2.BUILD",
    sha256 = "42f3d391dbf07b65a52d3d9eed8038ecd9fae53cf4e0e44e2adb95d0cd433b53",
    strip_prefix = "FFmpeg-n4.2.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/FFmpeg/FFmpeg/archive/n4.2.4.tar.gz",
        "https://github.com/FFmpeg/FFmpeg/archive/n4.2.4.tar.gz",
    ],
)

http_archive(
    name = "rules_python",
    sha256 = "c911dc70f62f507f3a361cbc21d6e0d502b91254382255309bc60b7a0f48de28",
    strip_prefix = "rules_python-38f86fb55b698c51e8510c807489c9f4e047480e",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_python/archive/38f86fb55b698c51e8510c807489c9f4e047480e.tar.gz",
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
    patch_cmds = [
        """sed -i.bak 's/"python",/"python3",/g' third_party/py/python_configure.bzl""",
        """sed -i.bak 's/PYTHONHASHSEED=0/PYTHONHASHSEED=0 python3/g' bazel/cython_library.bzl""",
    ],
    sha256 = "b956598d8cbe168b5ee717b5dafa56563eb5201a947856a6688bbeac9cac4e1f",
    strip_prefix = "grpc-b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
        "https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
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

http_archive(
    name = "build_bazel_rules_swift",
    # TODO: Remove once build_bazel_rules_swift support selectively choose the platform (macOS or Linux or Windows) to invoke the toolchain.
    patch_cmds = [
        "sed -i.bak 's/        _create_linux_toolchain/        print/g' swift/internal/swift_autoconfiguration.bzl",
    ],
    sha256 = "da799f591aed933f63575ef0fbf7b7a20a84363633f031fcd48c936cee771502",
    strip_prefix = "rules_swift-1b0fd91696928ce940bcc220f36c898694f10115",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_swift/archive/1b0fd91696928ce940bcc220f36c898694f10115.tar.gz",
        "https://github.com/bazelbuild/rules_swift/archive/1b0fd91696928ce940bcc220f36c898694f10115.tar.gz",
    ],
)

load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")

swift_rules_dependencies()

load("@build_bazel_rules_apple//apple:repositories.bzl", "apple_rules_dependencies")

apple_rules_dependencies()

load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")

apple_support_dependencies()

http_archive(
    name = "com_github_googleapis_google_cloud_cpp",
    patch_cmds = [
        """sed -i.bak 's/CURL\\* m/CURLM* m/g' google/cloud/storage/internal/curl_handle_factory.cc""",
    ],
    repo_mapping = {
        "@com_github_curl_curl": "@curl",
        "@com_github_nlohmann_json": "@nlohmann_json_lib",
    },
    sha256 = "14bf9bf97431b890e0ae5dca8f8904841d4883b8596a7108a42f5700ae58d711",
    strip_prefix = "google-cloud-cpp-1.21.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/googleapis/google-cloud-cpp/archive/v1.21.0.tar.gz",
        "https://github.com/googleapis/google-cloud-cpp/archive/v1.21.0.tar.gz",
    ],
)

http_archive(
    name = "nlohmann_json_lib",
    build_file = "//third_party:nlohmann_json.BUILD",
    sha256 = "c377963a95989270c943d522bfefe7b889ef5ed0e1e15d535fd6f6f16ed70732",
    strip_prefix = "json-3.4.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nlohmann/json/archive/v3.4.0.tar.gz",
        "https://github.com/nlohmann/json/archive/v3.4.0.tar.gz",
    ],
)

http_archive(
    name = "com_google_googleapis",
    build_file = "@com_github_googleapis_google_cloud_cpp//bazel:googleapis.BUILD",
    sha256 = "7ebab01b06c555f4b6514453dc3e1667f810ef91d1d4d2d3aa29bb9fcb40a900",
    strip_prefix = "googleapis-541b1ded4abadcc38e8178680b0677f65594ea6f",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/googleapis/googleapis/archive/541b1ded4abadcc38e8178680b0677f65594ea6f.zip",
        "https://github.com/googleapis/googleapis/archive/541b1ded4abadcc38e8178680b0677f65594ea6f.zip",
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
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/2018-10-01.tar.gz",
        "https://github.com/google/re2/archive/2018-10-01.tar.gz",
    ],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "9bf1fe5182a604b4135edc1a425ae356c9ad15e9b23f9f12a02e80184c3a249c",
    strip_prefix = "googletest-release-1.8.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/googletest/archive/release-1.8.1.tar.gz",
        "https://github.com/google/googletest/archive/release-1.8.1.tar.gz",
    ],
)

http_archive(
    name = "libarchive",
    build_file = "//third_party:libarchive.BUILD",
    sha256 = "720da414e7aebb255fcdaee106894e4d30e2472ac1390c2c15b70c84c7479658",
    strip_prefix = "libarchive-3.3.3",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/libarchive/libarchive/archive/v3.3.3.tar.gz",
        "https://github.com/libarchive/libarchive/archive/v3.3.3.tar.gz",
    ],
)

http_archive(
    name = "libexpat",
    build_file = "//third_party:libexpat.BUILD",
    sha256 = "574499cba22a599393e28d99ecfa1e7fc85be7d6651d543045244d5b561cb7ff",
    strip_prefix = "libexpat-R_2_2_6/expat",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/libexpat/libexpat/archive/R_2_2_6.tar.gz",
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
    sha256 = "096968a363b2374f7450a3c65f3cc0b50561204a8da7bc03a2c39e080febd6e1",
    strip_prefix = "apr-1.6.5",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/apr/archive/1.6.5.tar.gz",
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
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/apr-util/archive/1.6.1.tar.gz",
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
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/michaelrsweet/mxml/archive/v2.12.tar.gz",
        "https://github.com/michaelrsweet/mxml/archive/v2.12.tar.gz",
    ],
)

http_archive(
    name = "aliyun_oss_c_sdk",
    build_file = "//third_party:oss_c_sdk.BUILD",
    sha256 = "6450d3970578c794b23e9e1645440c6f42f63be3f82383097660db5cf2fba685",
    strip_prefix = "aliyun-oss-c-sdk-3.7.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/aliyun/aliyun-oss-c-sdk/archive/3.7.0.tar.gz",
        "https://github.com/aliyun/aliyun-oss-c-sdk/archive/3.7.0.tar.gz",
    ],
)

http_archive(
    name = "jsoncpp_git",
    build_file = "//third_party:jsoncpp.BUILD",
    sha256 = "c49deac9e0933bcb7044f08516861a2d560988540b23de2ac1ad443b219afdb6",
    strip_prefix = "jsoncpp-1.8.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
        "https://github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
    ],
)

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "ae8c36ff6e565f674c7a3692d6a9ea1096e4c1ade497272c2108a810fb39acd2",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_go/releases/download/0.19.4/rules_go-0.19.4.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/0.19.4/rules_go-0.19.4.tar.gz",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "7fc87f4170011201b1690326e8c16c5d802836e3a0d617d8f75c3af2b23180c4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-gazelle/releases/download/0.18.2/bazel-gazelle-0.18.2.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/0.18.2/bazel-gazelle-0.18.2.tar.gz",
    ],
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
    name = "nucleus",
    build_file = "//third_party:nucleus.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:nucleus.patch",
    ],
    sha256 = "aa865d3509ba8f3527392303bd95a11f48f19e68197b3d1d0bae9fab004bee87",
    strip_prefix = "nucleus-0.4.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nucleus/archive/0.4.1.tar.gz",
        "https://github.com/google/nucleus/archive/0.4.1.tar.gz",
    ],
)

http_archive(
    name = "bzip2",
    build_file = "//third_party:bzip2.BUILD",
    sha256 = "ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269",
    strip_prefix = "bzip2-1.0.8",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz",
        "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz",
    ],
)

http_archive(
    name = "com_googlesource_code_cctz",
    sha256 = "4ee3497b413229083998dd4295fa070b47a7253d88a15306733a06bae15ce945",
    strip_prefix = "cctz-44541cf2b85ced2a6e5ad4276183a9812d1a54ab",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/cctz/archive/44541cf2b85ced2a6e5ad4276183a9812d1a54ab.zip",
        "https://github.com/google/cctz/archive/44541cf2b85ced2a6e5ad4276183a9812d1a54ab.zip",
    ],
)

# This is the 1.9 release of htslib.
http_archive(
    name = "htslib",
    build_file = "//third_party:htslib.BUILD",
    sha256 = "c4d3ae84014f8a80f5011521f391e917bc3b4f6ebd78e97f238472e95849ec14",
    strip_prefix = "htslib-1.9",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/samtools/htslib/archive/1.9.zip",
        "https://github.com/samtools/htslib/archive/1.9.zip",
    ],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "43c9b882fa921923bcba764453f4058d102bece35a37c9f6383c713004aacff1",
    strip_prefix = "rules_closure-9889e2348259a5aad7e805547c1a0cf311cfcd91",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",  # 2018-12-21
        "https://github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",  # 2018-12-21
    ],
)

# bazel_skylib is now a required dependency of protobuf_archive.
http_archive(
    name = "bazel_skylib",
    sha256 = "bbccf674aa441c266df9894182d80de104cabd19be98be002f6d478aaa31574d",
    strip_prefix = "bazel-skylib-2169ae1c374aab4a09aa90e65efe1a3aad4e279b",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/archive/2169ae1c374aab4a09aa90e65efe1a3aad4e279b.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/archive/2169ae1c374aab4a09aa90e65efe1a3aad4e279b.tar.gz",
    ],
)

http_archive(
    name = "com_grail_bazel_toolchain",
    sha256 = "9e6065ded4b7453143e1586d6819729a63cd233114b72bf85ff3435367b02c90",
    strip_prefix = "bazel-toolchain-edd07e96a2ecaa131af9234d6582875d980c0ac7",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grailbio/bazel-toolchain/archive/edd07e96a2ecaa131af9234d6582875d980c0ac7.tar.gz",
        "https://github.com/grailbio/bazel-toolchain/archive/edd07e96a2ecaa131af9234d6582875d980c0ac7.tar.gz",
    ],
)

load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "9.0.0",
)

http_archive(
    name = "nasm",
    build_file = "//third_party:nasm.BUILD",
    sha256 = "34fd26c70a277a9fdd54cb5ecf389badedaf48047b269d1008fbc819b24e80bc",
    strip_prefix = "nasm-2.14.02",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/www.nasm.us/pub/nasm/releasebuilds/2.14.02/nasm-2.14.02.tar.bz2",
        "https://mirror.sobukus.de/files/src/nasm/nasm-2.14.02.tar.bz2",
        "http://www.nasm.us/pub/nasm/releasebuilds/2.14.02/nasm-2.14.02.tar.bz2",
    ],
)

http_archive(
    name = "vorbis",
    build_file = "//third_party:vorbis.BUILD",
    sha256 = "43fc4bc34f13da15b8acfa72fd594678e214d1cab35fc51d3a54969a725464eb",
    strip_prefix = "vorbis-1.3.6",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/xiph/vorbis/archive/v1.3.6.tar.gz",
        "https://github.com/xiph/vorbis/archive/v1.3.6.tar.gz",
    ],
)

http_archive(
    name = "ogg",
    build_file = "//third_party:ogg.BUILD",
    patch_cmds = [
        "sed -i.bak 's/define _OS_TYPES_H/define _OS_TYPES_H\\'$'\\n''#include <stdint.h>/g' include/ogg/os_types.h",
    ],
    sha256 = "3da31a4eb31534b6f878914b7379b873c280e610649fe5c07935b3d137a828bc",
    strip_prefix = "ogg-1.3.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/xiph/ogg/archive/v1.3.4.tar.gz",
        "https://github.com/xiph/ogg/archive/v1.3.4.tar.gz",
    ],
)

http_archive(
    name = "flac",
    build_file = "//third_party:flac.BUILD",
    sha256 = "668cdeab898a7dd43cf84739f7e1f3ed6b35ece2ef9968a5c7079fe9adfe1689",
    strip_prefix = "flac-1.3.3",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/xiph/flac/archive/1.3.3.tar.gz",
        "https://github.com/xiph/flac/archive/1.3.3.tar.gz",
    ],
)

http_archive(
    name = "minimp3",
    build_file = "//third_party:minimp3.BUILD",
    sha256 = "09395758f4c964fb158875f3cc9b9a65f36e9f5b2a27fb10f99519a0a6aef664",
    strip_prefix = "minimp3-55da78cbeea5fb6757f8df672567714e1e8ca3e9",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/lieff/minimp3/archive/55da78cbeea5fb6757f8df672567714e1e8ca3e9.tar.gz",
        "https://github.com/lieff/minimp3/archive/55da78cbeea5fb6757f8df672567714e1e8ca3e9.tar.gz",
    ],
)

http_archive(
    name = "speexdsp",
    build_file = "//third_party:speexdsp.BUILD",
    sha256 = "d7032f607e8913c019b190c2bccc36ea73fc36718ee38b5cdfc4e4c0a04ce9a4",
    strip_prefix = "speexdsp-SpeexDSP-1.2.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/xiph/speexdsp/archive/SpeexDSP-1.2.0.tar.gz",
        "https://github.com/xiph/speexdsp/archive/SpeexDSP-1.2.0.tar.gz",
    ],
)

http_archive(
    name = "minimp4",
    build_file = "//third_party:minimp4.BUILD",
    sha256 = "2c9e176b2df3f72d9cb3bcd0959ebfc9da3efcedbea70fb945270c7bfa9e7758",
    strip_prefix = "minimp4-14d452e4fac71da38f5c02e211486144075f4ecb",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/lieff/minimp4/archive/14d452e4fac71da38f5c02e211486144075f4ecb.tar.gz",
        "https://github.com/lieff/minimp4/archive/14d452e4fac71da38f5c02e211486144075f4ecb.tar.gz",
    ],
)

http_archive(
    name = "postgresql",
    build_file = "//third_party:postgresql.BUILD",
    sha256 = "9868c1149a04bae1131533c5cbd1c46f9c077f834f6147abaef8791a7c91b1a1",
    strip_prefix = "postgresql-12.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/ftp.postgresql.org/pub/source/v12.1/postgresql-12.1.tar.gz",
        "https://ftp.postgresql.org/pub/source/v12.1/postgresql-12.1.tar.gz",
    ],
)

http_archive(
    name = "libavif",
    build_file = "//third_party:libavif.BUILD",
    sha256 = "a4ce03649c58ec9f3dc6ab2b7cf7d58474b149acf1e4c563be4081bad60ed2dd",
    strip_prefix = "libavif-0.7.3",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/AOMediaCodec/libavif/archive/v0.7.3.tar.gz",
        "https://github.com/AOMediaCodec/libavif/archive/v0.7.3.tar.gz",
    ],
)

http_archive(
    name = "dav1d",
    build_file = "//third_party:dav1d.BUILD",
    patch_cmds = [
        "mkdir -p include8/common",
        "sed 's/define DAV1D_COMMON_BITDEPTH_H/define DAV1D_COMMON_BITDEPTH_H\\'$'\\n''#define BITDEPTH 8/g' include/common/bitdepth.h > include8/common/bitdepth.h",
        "cat include/common/dump.h > include8/common/dump.h",
        "mkdir -p include16/common",
        "sed 's/define DAV1D_COMMON_BITDEPTH_H/define DAV1D_COMMON_BITDEPTH_H\\'$'\\n''#define BITDEPTH 16/g' include/common/bitdepth.h > include16/common/bitdepth.h",
        "cat include/common/dump.h > include16/common/dump.h",
    ],
    sha256 = "66c3e831a93f074290a72aad5da907e3763ecb092325f0250a841927b3d30ce3",
    strip_prefix = "dav1d-0.6.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/videolan/dav1d/archive/0.6.0.tar.gz",
        "https://github.com/videolan/dav1d/archive/0.6.0.tar.gz",
    ],
)

new_git_repository(
    name = "libgav1",
    build_file = "//third_party:libgav1.BUILD",
    commit = "6ab7d65a68350ed4ec6aaabfa18715b2d76a231c",
    remote = "https://chromium.googlesource.com/codecs/libgav1",
)

http_archive(
    name = "libjpeg_turbo",
    build_file = "//third_party:libjpeg_turbo.BUILD",
    sha256 = "7777c3c19762940cff42b3ba4d7cd5c52d1671b39a79532050c85efb99079064",
    strip_prefix = "libjpeg-turbo-2.0.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.4.tar.gz",
        "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.4.tar.gz",
    ],
)

http_archive(
    name = "libmemcached",
    build_file = "//third_party:libmemcached.BUILD",
    patch_cmds = [
        "sed -i.bak 's/LIBMEMCACHED_WITH_SASL_SUPPORT 1/LIBMEMCACHED_WITH_SASL_SUPPORT 0/' libmemcached-1.0/configure.h",
    ],
    sha256 = "e22c0bb032fde08f53de9ffbc5a128233041d9f33b5de022c0978a2149885f82",
    strip_prefix = "libmemcached-1.0.18",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/launchpad.net/libmemcached/1.0/1.0.18/+download/libmemcached-1.0.18.tar.gz",
        "https://launchpad.net/libmemcached/1.0/1.0.18/+download/libmemcached-1.0.18.tar.gz",
    ],
)

http_archive(
    name = "farmhash_archive",
    build_file = "//third_party:farmhash.BUILD",
    sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",
    strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
        "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
    ],
)

http_archive(
    name = "giflib",
    build_file = "//third_party:giflib.BUILD",
    sha256 = "31da5562f44c5f15d63340a09a4fd62b48c45620cd302f77a6d9acf0077879bd",
    strip_prefix = "giflib-5.2.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz",
        "https://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz",
    ],
)

http_archive(
    name = "hadoop",
    build_file = "//third_party:hadoop.BUILD",
    sha256 = "5fd5831b12b1e0999bd352d6cca11ef80f883c81ffa898e53c68d8fe8d170e9f",
    strip_prefix = "hadoop-3.3.0-src",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/downloads.apache.org/hadoop/common/hadoop-3.3.0/hadoop-3.3.0-src.tar.gz",
        "https://downloads.apache.org/hadoop/common/hadoop-3.3.0/hadoop-3.3.0-src.tar.gz",
    ],
)

http_archive(
    name = "dlfcn-win32",
    build_file = "//third_party:dlfcn-win32.BUILD",
    sha256 = "f18a412e84d8b701e61a78252411fe8c72587f52417c1ef21ca93604de1b9c55",
    strip_prefix = "dlfcn-win32-1.2.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/dlfcn-win32/dlfcn-win32/archive/v1.2.0.tar.gz",
        "https://github.com/dlfcn-win32/dlfcn-win32/archive/v1.2.0.tar.gz",
    ],
)

http_archive(
    name = "pulsar",
    build_file = "//third_party:pulsar.BUILD",
    patch_cmds = [
        "cp pulsar-common/src/main/proto/PulsarApi.proto pulsar-client-cpp/lib",
        "sed -i.bak 's/define PULSAR_DEFINES_H_/define PULSAR_DEFINES_H_\\'$'\\n''#if defined(_MSC_VER)\\'$'\\n''#include <Windows.h>\\'$'\\n''#undef ERROR\\'$'\\n''#endif/g' pulsar-client-cpp/include/pulsar/defines.h",
        "sed -i.bak 's/define LIB_ACKGROUPINGTRACKER_H_/define LIB_ACKGROUPINGTRACKER_H_\\'$'\\n''#include <pulsar\\/defines.h>/g' pulsar-client-cpp/lib/AckGroupingTracker.h",
    ],
    sha256 = "08f19ca6d6353751ff0661403b16b71425bf7ada3d8835a38e426ae303b0e385",
    strip_prefix = "pulsar-2.6.1",
    urls = [
        "https://github.com/apache/pulsar/archive/v2.6.1.tar.gz",
    ],
)

http_archive(
    name = "libmongoc",
    build_file = "//third_party:libmongoc.BUILD",
    patch_cmds = [
        "sed -i.bak 's/undef MONGOC_LOG_DOMAIN/undef MONGOC_LOG_DOMAIN\\'$'\\n''# define BIO_get_ssl(b,sslp)  BIO_ctrl(b,BIO_C_GET_SSL,0,(char *)(sslp))\\'$'\\n''# define BIO_do_handshake(b)  BIO_ctrl(b,BIO_C_DO_STATE_MACHINE,0,NULL)/g' src/libmongoc/src/mongoc/mongoc-stream-tls-openssl.c",
    ],
    sha256 = "0a722180e5b5c86c415b9256d753b2d5552901dc5d95c9f022072c3cd336887e",
    strip_prefix = "mongo-c-driver-1.16.2",
    urls = [
        "https://github.com/mongodb/mongo-c-driver/releases/download/1.16.2/mongo-c-driver-1.16.2.tar.gz",
    ],
)

http_archive(
    name = "tinyobjloader",
    build_file = "//third_party:tinyobjloader.BUILD",
    sha256 = "b8c972dfbbcef33d55554e7c9031abe7040795b67778ad3660a50afa7df6ec56",
    strip_prefix = "tinyobjloader-2.0.0rc8",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/tinyobjloader/tinyobjloader/archive/v2.0.0rc8.tar.gz",
        "https://github.com/tinyobjloader/tinyobjloader/archive/v2.0.0rc8.tar.gz",
    ],
)

http_archive(
    name = "liborc",
    build_file = "//third_party:liborc.BUILD",
    patch_cmds = [
        "tar -xzf c++/libs/libhdfspp/libhdfspp.tar.gz -C c++/libs/libhdfspp",
    ],
    sha256 = "abdffe48b8d2e7776c3b541ee2241401e49774941ca4a8c759e5d795daec8a45",
    strip_prefix = "orc-rel-release-1.6.7",
    urls = [
        "https://github.com/apache/orc/archive/refs/tags/rel/release-1.6.7.tar.gz",
    ],
)

http_archive(
    name = "xsimd",
    build_file = "//third_party:xsimd.BUILD",
    sha256 = "45337317c7f238fe0d64bb5d5418d264a427efc53400ddf8e6a964b6bcb31ce9",
    strip_prefix = "xsimd-7.5.0",
    urls = [
        "https://github.com/xtensor-stack/xsimd/archive/refs/tags/7.5.0.tar.gz",
    ],
)
