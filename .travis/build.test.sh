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

export BAZEL_VERSION=0.20.0 BAZEL_OS=linux

# Path to shared libraries for running pytest
export TFIO_DATAPATH="bazel-bin"

# Install bazel, display log only if error
curl -sOL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
chmod +x bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
./bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh 2>&1 > bazel-install.log || (cat bazel-install.log && false)
rm -rf bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh bazel-install.log

# Install needed repo
DEBIAN_FRONTEND=noninteractive apt-get -y -qq update
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install python python3 ffmpeg > /dev/null

# Install the latest version of pip (needed for google-cloud-pubsub)
curl -sOL https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
python get-pip.py
rm -rf get-pip.py

if [[ ! -z ${TENSORFLOW_INSTALL} ]]; then
  python3 -m pip install -q ${TENSORFLOW_INSTALL}
  python -m pip install -q ${TENSORFLOW_INSTALL}
fi

./configure.sh
bazel build \
  --noshow_progress \
  --noshow_loading_progress \
  --spawn_strategy standalone \
  --verbose_failures \
  --test_output=errors -- \
  //tensorflow_io/...

python3 -m pip install -q pytest google-cloud-pubsub boto3 pyarrow==0.11.1 pandas==0.19.2
python -m pip install -q pytest google-cloud-pubsub boto3 pyarrow==0.11.1 pandas==0.19.2

python3 -m pytest tests
python -m pytest tests
