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
# docker run -i -t --rm -v $PWD:/v -w /v --net=host ubuntu:14.04 /v/.travis/python.release.sh

export BAZEL_VERSION=0.20.0 BAZEL_OS=linux

DEBIAN_FRONTEND=noninteractive apt-get -y -qq update
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install \
  software-properties-common > /dev/null

DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa

DEBIAN_FRONTEND=noninteractive apt-get -y -qq update
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install \
  gcc g++ make patch \
  python \
  python3 \
  python3.5 \
  python3.6 \
  unzip \
  curl > /dev/null

curl -sOL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
chmod +x bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh

# Install bazel, display log only if error
./bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh 2>&1 > bazel-install.log || (cat bazel-install.log && false)
rm -rf bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
rm -rf bazel-install.log

curl -OL https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.bz2
tar xfa patchelf-0.9.tar.bz2
(cd patchelf-0.9 && ./configure --prefix=/usr && make && make install)
rm -rf patchelf-0.9*
curl -sOL https://bootstrap.pypa.io/get-pip.py
python3.6 get-pip.py
python3.5 get-pip.py
python3 get-pip.py
python get-pip.py
rm -rf get-pip.py
python3 -m pip install -q auditwheel==1.5.0
python3 -m pip install -q wheel==0.31.1

if [[ ! -z ${TENSORFLOW_INSTALL} ]]; then
  python -m pip install -q ${TENSORFLOW_INSTALL}
fi

./configure.sh
bazel build \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors -- \
  //tensorflow_io/...

python setup.py --data bazel-bin -q bdist_wheel "$@"
python3 setup.py --data bazel-bin -q bdist_wheel "$@"
python3.5 setup.py --data bazel-bin -q bdist_wheel "$@"
python3.6 setup.py --data bazel-bin -q bdist_wheel "$@"
for f in dist/*.whl; do
  auditwheel repair $f
done
