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

export TENSORFLOW_INSTALL="$(python3 setup.py --install-require)"

export BAZEL_OS=$(uname | tr '[:upper:]' '[:lower:]')
curl -sSOL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
bash -e bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
bazel version

python3 -m pip --version

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools
python3 -m pip --version

python3 -m pip install -q ${TENSORFLOW_INSTALL}

python3 tools/build/configure.py

cat .bazelrc

bazel build \
  --crosstool_top=//third_party/toolchains/gcc7_manylinux2010:toolchain \
  ${BAZEL_OPTIMIZATION} \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  -- //tensorflow_io/...  //tensorflow_io_gcs_filesystem/...

rm -rf build && mkdir -p build

cp -r bazel-bin/tensorflow_io  build/tensorflow_io
cp -r bazel-bin/tensorflow_io_gcs_filesystem  build/tensorflow_io_gcs_filesystem

exit 0
