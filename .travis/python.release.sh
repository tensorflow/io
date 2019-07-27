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
# docker run -i -t --rm -v $PWD:/v -w /v --net=host buildpack-deps:14.04 /v/.travis/python.release.sh

export TENSORFLOW_INSTALL="$(python setup.py --package-version)"
export PYTHON_VERSION="python2.7 python3.4 python3.5 python3.6"
if [[ "$#" -gt 0 ]]; then
    export TENSORFLOW_INSTALL="${1}"
    shift
    PYTHON_VERSION="$@"
fi

bash -x -e .travis/bazel.configure.sh "${TENSORFLOW_INSTALL}"
bash -x -e .travis/bazel.build.sh
echo bash -x -e .travis/wheel.configure.sh ${PYTHON_VERSION}
echo bash -x -e .travis/wheel.build.sh ${PYTHON_VERSION}
