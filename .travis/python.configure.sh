apt-get -y -qq update
apt-get -y -qq install $INSTALL_PACKAGES
# Update Python and Pip alias
echo PYTHON_VERSION="${PYTHON_VERSION}"
if [ "${PYTHON_VERSION}" != "2.7" ]; then
  echo CUSTOM_OP="${CUSTOM_OP}"
  if [ -n "${CUSTOM_OP}" ]; then
    if [ "${PYTHON_VERSION}" == "3.4" ]; then
      rm -f /usr/bin/python
      ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
    fi
    if [ "${PYTHON_VERSION}" == "3.5" ]; then
      curl -OL https://raw.githubusercontent.com/tensorflow/tensorflow/v1.12.0/tensorflow/tools/ci_build/install/install_python3.5_pip_packages.sh
      chmod +x install_python3.5_pip_packages.sh
      ./install_python3.5_pip_packages.sh >install_python3.5_pip_packages.log 2>&1
      rm install_python3.5_pip_packages.sh
      rm -f /usr/bin/python
      ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
    fi
    if [ "${PYTHON_VERSION}" == "3.6" ]; then
      curl -OL https://raw.githubusercontent.com/tensorflow/tensorflow/v1.12.0/tensorflow/tools/ci_build/install/install_python3.6_pip_packages.sh
      chmod +x install_python3.6_pip_packages.sh
      sed -i 's/apt-get update/apt-get -y -qq update/g' install_python3.6_pip_packages.sh
      sed -i 's/apt-get upgrade/apt-get -y -qq upgrade/g' install_python3.6_pip_packages.sh
      sed -i 's/apt-get install/apt-get -y -qq install/g' install_python3.6_pip_packages.sh
      ./install_python3.6_pip_packages.sh >install_python3.6_pip_packages.log 2>&1
      rm install_python3.6_pip_packages.sh
      rm -f /usr/bin/python
      ln -s /usr/local/bin/python${PYTHON_VERSION} /usr/bin/python
      rm -f /usr/local/bin/python
      ln -s /usr/local/bin/python${PYTHON_VERSION} /usr/local/bin/python
    fi
    rm -f /usr/local/bin/pip
    ln -s /usr/local/bin/pip${PYTHON_VERSION} /usr/local/bin/pip
  else
    rm -f /usr/bin/pip
    ln -s /usr/bin/pip3 /usr/bin/pip
    rm -f /usr/bin/python
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
  fi
fi
# Show gcc and python version in Travis CI
gcc -v
python --version
pip --version
# Install bazel
curl -OL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
chmod +x bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
./bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
rm -f bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
# Install tensorflow
pip install tensorflow==1.12.0
# Configure TensorFlow
./configure.sh
