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
# docker run -it -e BAZEL_VERSION=$BAZEL_VERSION --rm -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op bash -x /working_dir/.travis/python.release.sh <2.7|3.4|3.5|3.6>

if [[ -z "${1}" ]]; then
  echo "usage:" $0 "<2.7|3.4|3.5|3.6>"
  exit 1
fi

export PYTHON_VERSION=$1 
export BAZEL_VERSION=${BAZEL_VERSION}
export CUSTOM_OP=True
export BAZEL_OS=linux
.travis/python.configure.sh
bazel build --noshow_progress --noshow_loading_progress --spawn_strategy standalone --verbose_failures --test_output=errors build_pip_pkg

bazel-bin/build_pip_pkg artifacts

exit 0
