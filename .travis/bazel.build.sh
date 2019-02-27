set -e

bazel build \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  -- //tensorflow_io/...
