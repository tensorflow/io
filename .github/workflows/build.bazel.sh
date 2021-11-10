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


PYTHON=python3
if [[ $# == 1 ]]; then
    PYTHON=$1
fi
$PYTHON --version

export TENSORFLOW_INSTALL="$($PYTHON setup.py --install-require)"

export BAZEL_OS=$(uname | tr '[:upper:]' '[:lower:]')
export BAZEL_VERSION=$(cat .bazelversion)
export PYTHON_BIN_PATH=`which $PYTHON`
curl -sSOL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
bash -e bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
bazel version

$PYTHON -m pip --version

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install --upgrade setuptools
$PYTHON -m pip --version

$PYTHON -m pip install -q ${TENSORFLOW_INSTALL}

$PYTHON tools/build/configure.py

cat .bazelrc

bazel build \
  ${BAZEL_OPTIMIZATION} \
  -- //tensorflow_io/...  //tensorflow_io_gcs_filesystem/...

rm -rf build && mkdir -p build

cp -r bazel-bin/tensorflow_io  build/tensorflow_io
cp -r bazel-bin/tensorflow_io_gcs_filesystem  build/tensorflow_io_gcs_filesystem

chown -R $(id -nu):$(id -ng) build/tensorflow_io/
chown -R $(id -nu):$(id -ng) build/tensorflow_io_gcs_filesystem/
find build/tensorflow_io -name '*runfiles*' | xargs rm -rf
find build/tensorflow_io_gcs_filesystem -name '*runfiles*' | xargs rm -rf

exit 0
