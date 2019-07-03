set -e

if [[ $(uname) == "Darwin" ]]; then
# The -DGRPC_BAZEL_BUILD is needed because gRPC does not compile on macOS unless
# it is set.
  bazel build \
    --copt=-DGRPC_BAZEL_BUILD \
    --noshow_progress \
    --noshow_loading_progress \
    --verbose_failures \
    --test_output=errors \
    -- //tensorflow_io/...
else
  bazel build \
    --noshow_progress \
    --noshow_loading_progress \
    --verbose_failures \
    --test_output=errors \
    -- //tensorflow_io/...
fi