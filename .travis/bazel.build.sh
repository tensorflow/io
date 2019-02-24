set -e

if [[ $(uname) == "Darwin" ]]; then
  NOBUILD="-//tensorflow_io/ignite:all -//tensorflow_io/kafka:all -//tensorflow_io/kinesis:all"
fi

bazel build \
  --copt="-D_GLIBCXX_USE_CXX11_ABI=0" \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  -- //tensorflow_io/... $NOBUILD
