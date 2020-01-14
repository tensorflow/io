set -e

python3 --version

./configure.sh

cat .bazelrc

${BAZEL_PATH:=bazel} build -s --verbose_failures //tensorflow_io/core:python/ops/libtensorflow_io.so

exit $?
