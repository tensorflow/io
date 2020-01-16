set -e -x

python3 --version

export MSYS2_ARG_CONV_EXCL="//"
${BAZEL_PATH:=bazel} build -s --verbose_failures @zlib//:zlib
${BAZEL_PATH:=bazel} build -s --verbose_failures @curl//:curl
${BAZEL_PATH:=bazel} build -s --verbose_failures @com_github_azure_azure_storage_cpplite//:azure
${BAZEL_PATH:=bazel} build -s --verbose_failures //tensorflow_io/core:python/ops/libtensorflow_io.so

exit $?
