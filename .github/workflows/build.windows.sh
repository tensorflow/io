set -e

python3 --version

${BAZEL_PATH:=bazel} build -s --verbose_failures //tensorflow_io/...

exit $?
