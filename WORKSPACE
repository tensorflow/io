workspace(name = "org_tensorflow_io")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Note: zlib is placed earlier as tensorflow's zlib does not include unzip
http_archive(
    name = "zlib",
    build_file = "//third_party:zlib.BUILD",
    patch_cmds = ["""sed -i.bak '29i\\'$'\\n#include<zconf.h>\\n' contrib/minizip/crypt.h"""],
    sha256 = "b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30",
    strip_prefix = "zlib-1.2.13",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.13.tar.gz",
        "https://zlib.net/zlib-1.2.13.tar.gz",
    ],
)

# Note: snappy is placed earlier as tensorflow's snappy does not include snappy-c
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

# Note: boringssl is placed earlier as boringssl needs to be patched for mongodb
http_archive(
    name = "boringssl",
    patch_cmds = [
        """sed -i.bak 's/bio.c",/bio.c","src\\/decrepit\\/bio\\/base64_bio.c",/g' BUILD.generated.bzl""",
    ],
    sha256 = "a9c3b03657d507975a32732f04563132b4553c20747cec6dc04de475c8bdf29f",
    strip_prefix = "boringssl-80ca9f9f6ece29ab132cce4cf807a9465a18cfac",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/boringssl/archive/80ca9f9f6ece29ab132cce4cf807a9465a18cfac.tar.gz",
        "https://github.com/google/boringssl/archive/80ca9f9f6ece29ab132cce4cf807a9465a18cfac.tar.gz",
    ],
)

# Note google_cloud_cpp is placed earlier as tensorflow's version is older
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

# Note com_google_googleapis is placed earlier as we need to adjust switched_rules_by_language option
# Note we have to change one word in the field_behavior.proto so it compiles on WINDOWS
# for more infor please refer to https://github.com/protocolbuffers/protobuf/issues/7076
# Because of a bug in protocol buffers (protocolbuffers/protobuf#7076), new versions of this project
# fail to compile on Windows. The problem hinges on OPTIONAL being defined as an empty string under
# Windows. This makes the preprocessor remove every mention of OPTIONAL from the code, which causes
# compilation failures. This temporary workaround renames the name of the protobuf value OPTIONAL to
# OPIONAL. This should be safe as it does not affect the generated protobufs.
http_archive(
    name = "com_google_googleapis",
    build_file = "@com_github_googleapis_google_cloud_cpp//bazel:googleapis.BUILD",
    patch_cmds = [
        """sed -i.bak 's/OPTIONAL/OPIONAL/g' google/api/field_behavior.proto""",
        """sed -i.bak 's/OPTIONAL/OPIONAL/g' google/pubsub/v1beta2/pubsub.proto""",
        """sed -i.bak 's/OPTIONAL/OPIONAL/g' google/pubsub/v1/pubsub.proto""",
    ],
    sha256 = "249d83abc5d50bf372c35c49d77f900bff022b2c21eb73aa8da1458b6ac401fc",
    strip_prefix = "googleapis-6b3fdcea8bc5398be4e7e9930c693f0ea09316a0",
    urls = [
        "https://github.com/googleapis/googleapis/archive/6b3fdcea8bc5398be4e7e9930c693f0ea09316a0.tar.gz",
    ],
)

load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")

# Configure @com_google_googleapis to only compile C++ and gRPC:
switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,  # C++ support is only "Partially implemented", roll our own.
    grpc = True,
)

http_archive(
    name = "org_tensorflow",
    sha256 = "c729e56efc945c6df08efe5c9f5b8b89329c7c91b8f40ad2bb3e13900bd4876d",
    strip_prefix = "tensorflow-2.16.1",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.16.1.tar.gz",
    ],
    patch_cmds = [
        """sed -i.bak 's/cython-3.0.3/cython-3.0.0a11/g' tensorflow/workspace2.bzl""",
        """sed -i.bak 's/3.0.3.tar.gz/3.0.0a11.tar.gz/g' tensorflow/workspace2.bzl""",
        """sed -i.bak 's/0c2eae8a4ceab7955be1e11a4ddc5dcc3aa06ce22ad594262f1555b9d10667f0/08dbdb6aa003f03e65879de8f899f87c8c718cd874a31ae9c29f8726da2f5ab0/g' tensorflow/workspace2.bzl""",
    ],
)

http_archive(
    name = "rules_python",
    sha256 = "84aec9e21cc56fbc7f1335035a71c850d1b9b5cc6ff497306f84cced9a769841",
    strip_prefix = "rules_python-0.23.1",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.23.1/rules_python-0.23.1.tar.gz",
)

load("@rules_python//python:repositories.bzl", "python_register_toolchains")
load(
    "@org_tensorflow//tensorflow/tools/toolchains/python:python_repo.bzl",
    "python_repository",
)

python_repository(name = "python_version_repo")

load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")

python_register_toolchains(
    name = "python",
    ignore_root_user_error = True,
    python_version = HERMETIC_PYTHON_VERSION,
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("//third_party/toolchains/tf:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

http_archive(
    name = "lmdb",
    build_file = "//third_party:lmdb.BUILD",
    sha256 = "22054926b426c66d8f2bc22071365df6e35f3aacf19ad943bc6167d4cae3bebb",
    strip_prefix = "lmdb-LMDB_0.9.29/libraries/liblmdb",
    urls = [
        "https://github.com/LMDB/lmdb/archive/refs/tags/LMDB_0.9.29.tar.gz",
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
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    sha256 = "57e13c62f27b710e1de54fd30faed612aefa22aa41fa2c0c3bacd204dd18a8f3",
    strip_prefix = "arrow-apache-arrow-7.0.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/arrow/archive/apache-arrow-7.0.0.tar.gz",
        "https://github.com/apache/arrow/archive/apache-arrow-7.0.0.tar.gz",
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
        """sed -i.bak 's/__attribute__((visibility("default")))//g' aws-cpp-sdk-core/include/aws/core/external/tinyxml2/tinyxml2.h """,
    ],
    sha256 = "ae1cb22225b1f47eee351c0064be5e87676bf7090bb9ad19888bea0dab0e2749",
    strip_prefix = "aws-sdk-cpp-1.8.187",
    urls = [
        "https://github.com/aws/aws-sdk-cpp/archive/1.8.187.tar.gz",
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
    name = "bzip2",
    build_file = "//third_party:bzip2.BUILD",
    sha256 = "329e4eb98f6af8d39da05cb51bccec88ae015eac99a42b1ee04dec0af7f4b957",
    strip_prefix = "bzip2-bzip2-1.0.8",
    urls = [
        "https://gitlab.com/bzip2/bzip2/-/archive/bzip2-1.0.8/bzip2-bzip2-1.0.8.tar.gz",
    ],
)

http_archive(
    name = "libxml_archive",
    build_file = "@//third_party:libxml.BUILD",
    patch_cmds_win = [
        """sed -i 's/define LIBXML_ICONV_ENABLED/undef LIBXML_ICONV_ENABLED/g' include/libxml/xmlversion.h""",
        """sed -i 's/define LIBXML_LZMA_ENABLED/undef LIBXML_LZMA_ENABLED/g' include/libxml/xmlversion.h""",
    ],
    sha256 = "f63c5e7d30362ed28b38bfa1ac6313f9a80230720b7fb6c80575eeab3ff5900c",
    strip_prefix = "libxml2-2.9.7",
    urls = [
        "https://mirror.bazel.build/xmlsoft.org/sources/libxml2-2.9.7.tar.gz",
        "http://xmlsoft.org/sources/libxml2-2.9.7.tar.gz",
    ],
)

http_archive(
    name = "com_github_azure_azure_sdk_for_cpp",
    build_file = "//third_party:azure.BUILD",
    patch_cmds = [
        # patch can be removed once https://github.com/googleapis/google-cloud-cpp/issues/7462 is fixed
        """sed -i.bak 's/curl_easy_init();/curl_easy_init();curl_easy_setopt(newHandle, CURLOPT_NOSIGNAL, 1);/g' sdk/core/azure-core/src/http/curl/curl.cpp """,
        """sed -i.bak 's/include <windows.h>/include <windows.h>\\'$'\\n''#include <wincrypt.h>/g' sdk/core/azure-core/src/base64.cpp """,
    ],
    sha256 = "ec9cb17cab24e940895eb2249c096f500f69383edfa66b20cb6456414767ce99",
    strip_prefix = "azure-sdk-for-cpp-9dac89c67564c64748ebb72b9de55c548db51eff",
    urls = [
        "https://github.com/Azure/azure-sdk-for-cpp/archive/9dac89c67564c64748ebb72b9de55c548db51eff.tar.gz",
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

http_archive(
    name = "dcmtk",
    build_file = "//third_party:dcmtk.BUILD",
    sha256 = "fb1434c421d4cc5d391fe37d05f4a4a4267aab16af9826474a6ef366952a11cc",
    strip_prefix = "dcmtk-5fba853b6f7c13b02bed28bd9f7d3f450e4c72bb",
    urls = [
        "https://github.com/DCMTK/dcmtk/archive/5fba853b6f7c13b02bed28bd9f7d3f450e4c72bb.tar.gz",
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
    name = "double-conversion",
    sha256 = "a63ecb93182134ba4293fd5f22d6e08ca417caafa244afaa751cbfddf6415b13",
    strip_prefix = "double-conversion-3.1.5",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/double-conversion/archive/v3.1.5.tar.gz",
        "https://github.com/google/double-conversion/archive/v3.1.5.tar.gz",
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
    name = "fmjpeg2koj",
    build_file = "//third_party:fmjpeg2koj.BUILD",
    sha256 = "c5b203ec580cab6fbd92c24712e987e960bda45638d4c2061d0b15d5d520ab42",
    strip_prefix = "fmjpeg2koj-1.0.3",
    urls = [
        "https://github.com/DraconPern/fmjpeg2koj/archive/refs/tags/v1.0.3.tar.gz",
    ],
)

http_archive(
    name = "freetype",
    build_file = "//third_party:freetype.BUILD",
    patch_cmds = [
        """sed -i.bak 's/__attribute__(( visibility( "default" ) ))//g' include/freetype/config/ftconfig.h """,
    ],
    sha256 = "3a60d391fd579440561bf0e7f31af2222bc610ad6ce4d9d7bd2165bca8669110",
    strip_prefix = "freetype-2.10.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/download.savannah.gnu.org/releases/freetype/freetype-2.10.1.tar.gz",
        "https://download.savannah.gnu.org/releases/freetype/freetype-2.10.1.tar.gz",
    ],
)

http_archive(
    name = "hadoop",
    build_file = "//third_party:hadoop.BUILD",
    sha256 = "fa9d0587d06c36838e778081bcf8271a9c63060af00b3bf456423c1777a62043",
    strip_prefix = "hadoop-rel-release-3.3.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/apache/hadoop/archive/refs/tags/rel/release-3.3.0.tar.gz",
        "https://github.com/apache/hadoop/archive/refs/tags/rel/release-3.3.0.tar.gz",
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
    sha256 = "1e4299da5a3eca49cc3acab60600d0d7c0cda2de46d662ca14fadf5ab68a8c4f",
    strip_prefix = "apr-util-1.6.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/apr-util/archive/1.6.1.tar.gz",
        "https://github.com/apache/apr-util/archive/1.6.1.tar.gz",
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
    name = "libexpat",
    build_file = "//third_party:libexpat.BUILD",
    sha256 = "574499cba22a599393e28d99ecfa1e7fc85be7d6651d543045244d5b561cb7ff",
    strip_prefix = "libexpat-R_2_2_6/expat",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/libexpat/libexpat/archive/R_2_2_6.tar.gz",
        "http://github.com/libexpat/libexpat/archive/R_2_2_6.tar.gz",
    ],
)

new_git_repository(
    name = "libgav1",
    build_file = "//third_party:libgav1.BUILD",
    commit = "07a59c59d4d180d67ea0ae5269e3c092c87286e5",
    remote = "https://chromium.googlesource.com/codecs/libgav1",
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
    name = "libmongoc",
    build_file = "//third_party:libmongoc.BUILD",
    patch_cmds = [
        "sed -i.bak 's/undef MONGOC_LOG_DOMAIN/undef MONGOC_LOG_DOMAIN\\'$'\\n''# define BIO_get_ssl(b,sslp)  BIO_ctrl(b,BIO_C_GET_SSL,0,(char *)(sslp))\\'$'\\n''# define BIO_do_handshake(b)  BIO_ctrl(b,BIO_C_DO_STATE_MACHINE,0,NULL)/g' src/libmongoc/src/mongoc/mongoc-stream-tls-openssl.c",
        """sed -i.bak 's/__attribute__ ((visibility ("default")))//g' src/libmongoc/src/mongoc/mongoc-macros.h """,
        """sed -i.bak 's/__attribute__ ((visibility ("default")))//g' src/libbson/src/bson/bson-macros.h """,
    ],
    sha256 = "0a722180e5b5c86c415b9256d753b2d5552901dc5d95c9f022072c3cd336887e",
    strip_prefix = "mongo-c-driver-1.16.2",
    urls = [
        "https://github.com/mongodb/mongo-c-driver/releases/download/1.16.2/mongo-c-driver-1.16.2.tar.gz",
    ],
)

http_archive(
    name = "liborc",
    build_file = "//third_party:liborc.BUILD",
    patch_cmds = [
        "tar -xzf c++/libs/libhdfspp/libhdfspp.tar.gz -C c++/libs/libhdfspp",
    ],
    sha256 = "39d983f4c7feb8ea1e8ab8e3e53e9afc643282b7a500b3a93c91aa6490f65c17",
    strip_prefix = "orc-rel-release-1.6.14",
    urls = [
        "https://github.com/apache/orc/archive/refs/tags/rel/release-1.6.14.tar.gz",
    ],
)

http_archive(
    name = "libtiff",
    build_file = "//third_party:libtiff.BUILD",
    sha256 = "0e46e5acb087ce7d1ac53cf4f56a09b221537fc86dfc5daaad1c2e89e1b37ac8",
    strip_prefix = "tiff-4.3.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/download.osgeo.org/libtiff/tiff-4.3.0.tar.gz",
        "https://download.osgeo.org/libtiff/tiff-4.3.0.tar.gz",
    ],
)

http_archive(
    name = "libwebp",
    build_file = "//third_party:libwebp.BUILD",
    sha256 = "01bcde6a40a602294994050b81df379d71c40b7e39c819c024d079b3c56307f4",
    strip_prefix = "libwebp-1.2.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/webmproject/libwebp/archive/v1.2.1.tar.gz",
        "https://github.com/webmproject/libwebp/archive/v1.2.1.tar.gz",
    ],
)

new_git_repository(
    name = "libyuv",
    build_file = "//third_party:libyuv.BUILD",
    commit = "7f00d67d7c279f13b73d3be9c2d85873a7e2fbaf",
    remote = "https://chromium.googlesource.com/libyuv/libyuv",
)

http_archive(
    name = "lz4",
    build_file = "//third_party:lz4.BUILD",
    patch_cmds = [
        """sed -i.bak 's/__attribute__ ((__visibility__ ("default")))//g' lib/lz4frame.h """,
    ],
    sha256 = "658ba6191fa44c92280d4aa2c271b0f4fbc0e34d249578dd05e50e76d0e5efcc",
    strip_prefix = "lz4-1.9.2",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/lz4/lz4/archive/v1.9.2.tar.gz",
        "https://github.com/lz4/lz4/archive/v1.9.2.tar.gz",
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
    name = "openexr",
    build_file = "//third_party:openexr.BUILD",
    sha256 = "4904c5ea7914a58f60a5e2fbc397be67e7a25c380d7d07c1c31a3eefff1c92f1",
    strip_prefix = "openexr-2.4.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/openexr/openexr/archive/v2.4.0.tar.gz",
        "https://github.com/openexr/openexr/archive/v2.4.0.tar.gz",
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
    name = "proj",
    build_file = "//third_party:proj.BUILD",
    patch_cmds = [
        """sed -i.bak 's/include <sqlite3.h>/include "sqlite3.h"/' src/sqlite3_utils.hpp""",
        """sed -i.bak 's/include <sqlite3.h>/include "sqlite3.h"/' src/iso19111/factory.cpp""",
        """sed -i.bak 's/include <sqlite3.h>/include "sqlite3.h"/' src/proj_json_streaming_writer.cpp""",
    ],
    sha256 = "f0c88738b1bd3b65a217734b56a763988ea1ca4c779e39d9d9a8b5878888cd6f",
    strip_prefix = "proj-8.0.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/OSGeo/PROJ/releases/download/7.2.1/proj-8.0.0.zip",
        "https://github.com/OSGeo/PROJ/releases/download/8.0.0/proj-8.0.0.zip",
    ],
)

http_archive(
    name = "pulsar",
    build_file = "//third_party:pulsar.BUILD",
    sha256 = "be97723dbba43045506f877cbc7600d2efe74264eace980933ae42b387069bc3",
    strip_prefix = "pulsar-client-cpp-3.3.0",
    urls = [
        "https://github.com/apache/pulsar-client-cpp/archive/refs/tags/v3.3.0.tar.gz",
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
    name = "xsimd",
    build_file = "//third_party:xsimd.BUILD",
    sha256 = "21b4700e9ef70f6c9a86952047efd8272317df4e6fee35963de9394fd9c5677f",
    strip_prefix = "xsimd-8.0.1",
    urls = [
        "https://github.com/xtensor-stack/xsimd/archive/refs/tags/8.0.1.tar.gz",
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
    name = "zstd",
    build_file = "//third_party:zstd.BUILD",
    sha256 = "a364f5162c7d1a455cc915e8e3cf5f4bd8b75d09bc0f53965b0c9ca1383c52c8",
    strip_prefix = "zstd-1.4.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/facebook/zstd/archive/v1.4.4.tar.gz",
        "https://github.com/facebook/zstd/archive/v1.4.4.tar.gz",
    ],
)

# Needed for llvm_toolchain and Golang
http_archive(
    name = "com_grail_bazel_toolchain",
    sha256 = "4b7999c1fa2c3117bb21651e3c155b152e44ae67b2c311214883d4707dbe183f",
    strip_prefix = "toolchains_llvm-edd07e96a2ecaa131af9234d6582875d980c0ac7",
    urls = [
        "https://github.com/grailbio/bazel-toolchain/archive/edd07e96a2ecaa131af9234d6582875d980c0ac7.tar.gz",
    ],
)

load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "9.0.0",
)

# Golang related ruls, consider removal after switching to C++/C client for prometheus
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
