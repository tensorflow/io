set -e

python3 --version

./configure.sh

cat .bazelrc

${BAZEL_PATH:=bazel} build -s --verbose_failures //tensorflow_io/...

exit $?
