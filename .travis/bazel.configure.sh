set -e

export TENSORFLOW_INSTALL="${1}"

gcc -v
python --version
export BAZEL_VERSION=0.24.1 BAZEL_OS=$(uname | tr '[:upper:]' '[:lower:]')
curl -sSOL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
bash -e bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh 2>&1 > bazel-install.log || (cat bazel-install.log && false)
bazel version
curl -sSOL https://bootstrap.pypa.io/get-pip.py
python get-pip.py -q
python -m pip --version
if [[ $(uname) == "Darwin" ]]; then
  python -m pip install -q -U matplotlib numpy --ignore-installed six
else
  apt-get -y -qq install git
fi
python -m pip install -q --ignore-installed six "${TENSORFLOW_INSTALL}"
python -c 'import tensorflow as tf; print(tf.version.VERSION)'
bash -e -x configure.sh
