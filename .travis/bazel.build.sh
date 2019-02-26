set -e

if [[ $(uname) == "Darwin" ]]; then
  NOBUILD="-//tensorflow_io/ignite:all -//tensorflow_io/kafka:all -//tensorflow_io/kinesis:all"
fi

bazel build \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  -- //tensorflow_io/... $NOBUILD
