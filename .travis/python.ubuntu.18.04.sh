set -x -e

# Test on Travis inside Ubuntu18.04 docker image
apt-get -y -qq update
apt-get -y -qq install python${PYTHON_VERSION}-pip curl wget unzip make
if [[ "" != "${PYTHON_VERSION}" ]]; then
  ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
  ln -s /usr/bin/pip${PYTHON_VERSION} /usr/bin/pip
fi
# Show gcc and python version in Travis CI
gcc -v
python --version
# Install bazel
URL="https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh"
wget -O install.sh "${URL}"
chmod +x install.sh
./install.sh
rm -f install.sh
# Configure TensorFlow
./configure.sh
