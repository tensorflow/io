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
# docker run -i -t --rm -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:custom-op bash -x /working_dir/.travis/python.release.sh <2.7|3.4|3.5|3.6>
# Nightly:
# export TENSORFLOW_IO_VERSION=0.3.0.dev$(date '+%Y%m%d%H%M%S')
# docker run -i -t --rm -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:custom-op bash -x /working_dir/.travis/python.release.sh <2.7|3.4|3.5|3.6> <tensorflow-version> <project-name> $TENSORFLOW_IO_VERSION

if [[ -z "${1}" ]]; then
  echo "usage:" $0 "<2.7|3.4|3.5|3.6>"
  exit 1
fi

BAZEL_VERSION=0.20.0
PYTHON_VERSION=${1}
shift
TENSORFLOW_VERSION=1.12.0
if [[ ! -z ${1} ]]; then
  TENSORFLOW_VERSION=${1}
  shift
fi

.travis/bazel.install.sh linux ${BAZEL_VERSION}

.travis/python.install.sh ${PYTHON_VERSION}

pip install -q tensorflow==${TENSORFLOW_VERSION}
./configure.sh
bazel test --noshow_progress --noshow_loading_progress --spawn_strategy standalone --verbose_failures --test_output=errors -- //tensorflow_io/...
bazel build --noshow_progress --noshow_loading_progress --spawn_strategy standalone --verbose_failures --test_output=errors build_pip_pkg
bazel-bin/build_pip_pkg artifacts

exit 0
