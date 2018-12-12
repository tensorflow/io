URL="https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh"
wget -O install.sh "${URL}"
chmod +x install.sh
./install.sh --user
rm -f install.sh
./configure.sh

