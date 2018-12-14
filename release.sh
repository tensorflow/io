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
set -e
set -x

# Release:
# docker run -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op bash -x /working_dir/release.sh <2.7|3.4|3.5|3.6>

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

if [[ -z "${1}" ]]; then
  echo "usage:" $0 "<2.7|3.4|3.5|3.6>"
  exit 1
fi

PYTHON_VERSION=$1 

if [[ "3.5" == "${PYTHON_VERSION}" ]] || [ "3.6" == "${PYTHON_VERSION}" ]; then
  # fkrull/deadsnakes is for Python3.5
  add-apt-repository -y ppa:fkrull/deadsnakes
  apt-get update

  apt-get install -y --no-install-recommends python${PYTHON_VERSION} libpython${PYTHON_VERSION}-dev
  wget -q https://bootstrap.pypa.io/get-pip.py
  python${PYTHON_VERSION} get-pip.py
  rm -f get-pip.py
  pip${PYTHON_VERSION} install --upgrade pip
fi

rm -f /usr/bin/python /usr/bin/pip
ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
ln -s /usr/bin/pip${PYTHON_VERSION} /usr/bin/pip

./configure.sh

bazel build build_pip_pkg

bazel-bin/build_pip_pkg artifacts

exit 0
