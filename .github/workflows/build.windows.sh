set -e -x

python3 --version

export MSYS2_ARG_CONV_EXCL="//"
${BAZEL_PATH:=bazel} build -s --verbose_failures //tensorflow_io/core:python/ops/libtensorflow_io.so

exit $?
