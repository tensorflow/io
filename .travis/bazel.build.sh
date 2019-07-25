set -e

args=()
if [[ $(uname) == "Darwin" ]]; then 
  # The -DGRPC_BAZEL_BUILD is needed because gRPC does not compile on macOS unless
  # it is set.
  args+=(--copt=-DGRPC_BAZEL_BUILD)
fi
if [[ "${BAZEL_CACHE}" != "" ]]; then
  args+=(--disk_cache=${BAZEL_CACHE})
fi

bazel build \
  "${args[@]}" \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  //tensorflow_io/...
