#!/usr/bin/env bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -e -x

# Release:
# docker run -i -t --rm -v $PWD:/v -w /v --net=host ubuntu:16.04 /v/.travis/python3.7+.release.sh


apt-get -y -qq update && apt-get -y -qq install \
    software-properties-common \
    gcc g++ make patch \
    unzip curl patchelf

add-apt-repository -y ppa:deadsnakes/ppa

apt-get -y -qq update

curl -sSOL https://bootstrap.pypa.io/get-pip.py

export PYTHON_VERSION="python3.7"
if [[ "$#" -gt 0 ]]; then
    export PYTHON_VERSION="${1}"
    shift
fi

apt-get -y -qq update && apt-get -y -qq install python $PYTHON_VERSION
python get-pip.py -q
python -m pip --version
python -m pip install -q grpcio-tools

$PYTHON_VERSION get-pip.py -q
$PYTHON_VERSION -m pip --version

export TENSORFLOW_INSTALL="$(${PYTHON_VERSION} setup.py --package-version)"
if [[ "$#" -gt 0 ]]; then
    export TENSORFLOW_INSTALL="${1}"
    shift
fi
export BAZEL_VERSION=0.24.1 BAZEL_OS=$(uname | tr '[:upper:]' '[:lower:]')



curl -sSOL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
bash -e bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh 2>&1 > bazel-install.log || (cat bazel-install.log && false)
bazel version

$PYTHON_VERSION -m pip install -q "${TENSORFLOW_INSTALL}"
$PYTHON_VERSION -c 'import tensorflow as tf; print(tf.version.VERSION)'
$PYTHON_VERSION config_helper.py

bash -x -e .travis/bazel.build.sh


if [[ "$1" == "--"* ]]; then
  VERSION_CHOICE=$1
  VERSION_NUMBER=$2
  shift
  shift
fi

$PYTHON_VERSION setup.py --data bazel-bin -q bdist_wheel $VERSION_CHOICE $VERSION_NUMBER
ls dist/*

$PYTHON_VERSION -m pip install -q wheel==0.31.1
$PYTHON_VERSION -m pip install -q auditwheel==1.5.0
auditwheel --version
for f in dist/*.whl; do
    auditwheel repair $f
done
ls wheelhouse/*

