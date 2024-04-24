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
set -x


PYTHON=python3
if [[ $# == 1 ]]; then
    PYTHON=$1
fi
$PYTHON --version


export PYTHON_BIN_PATH=`which $PYTHON`

if [[ $(uname) == "Linux" ]]; then
  curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
  mv bazelisk-linux-amd64 /usr/local/bin/bazel
  chmod +x /usr/local/bin/bazel
  cp -f $(which $PYTHON) /usr/bin/python3
fi

bazel version

$PYTHON -m pip --version

if [[ $(uname) == "Darwin" && $(uname -m) == "arm64" ]]; then
$PYTHON -m pip install --upgrade --break-system-packages pip
$PYTHON -m pip install --upgrade --break-system-packages setuptools
$PYTHON -m pip --version

export TENSORFLOW_INSTALL="$($PYTHON setup.py --install-require)"

$PYTHON -m pip install --break-system-packages ${TENSORFLOW_INSTALL}
$PYTHON -m pip install --break-system-packages  "urllib3 <2"
$PYTHON -m pip uninstall --break-system-packages -y tensorflow-io-gcs-filesystem
else
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install --upgrade setuptools
$PYTHON -m pip --version

export TENSORFLOW_INSTALL="$($PYTHON setup.py --install-require)"

$PYTHON -m pip install -q ${TENSORFLOW_INSTALL}
$PYTHON -m pip install -q "urllib3 <2"
$PYTHON -m pip uninstall -y tensorflow-io-gcs-filesystem
fi


$PYTHON tools/build/configure.py

cat .bazelrc

if [[ $(uname -m) != "arm64" && $(uname) == "Darwin" ]]; then

bazel build \
  ${BAZEL_OPTIMIZATION} \
  -- //tensorflow_io:python/ops/libtensorflow_io.so //tensorflow_io:python/ops/libtensorflow_io_plugins.so //tensorflow_io_gcs_filesystem/...

elif [[ $(uname -m) == "arm64" && $(uname) == "Darwin" ]]; then

bazel build \
  ${BAZEL_OPTIMIZATION} \
  -- //tensorflow_io_gcs_filesystem/... //tensorflow_io:python/ops/libtensorflow_io.so //tensorflow_io:python/ops/libtensorflow_io_plugins.so

else

bazel build \
  ${BAZEL_OPTIMIZATION} \
  -- //tensorflow_io/...  //tensorflow_io_gcs_filesystem/...

fi

rm -rf build && mkdir -p build

if [[ $(uname) == "Linux" ]]; then
cp -r -L bazel-bin/tensorflow_io  build/tensorflow_io
cp -r -L bazel-bin/tensorflow_io_gcs_filesystem  build/tensorflow_io_gcs_filesystem
else
cp -r bazel-bin/tensorflow_io  build/tensorflow_io
cp -r bazel-bin/tensorflow_io_gcs_filesystem  build/tensorflow_io_gcs_filesystem
fi

echo chown -R $(id -nu):$(id -ng) build/tensorflow_io/
echo chown -R $(id -nu):$(id -ng) build/tensorflow_io_gcs_filesystem/
find build/tensorflow_io -name '*runfiles*' | xargs rm -rf
find build/tensorflow_io_gcs_filesystem -name '*runfiles*' | xargs rm -rf

exit 0
