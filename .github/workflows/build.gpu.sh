#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
# Make sure we're in the project root path.
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
ROOT_DIR=$( cd "$SCRIPT_DIR/../.." && pwd -P )
if [[ ! -d "tensorflow_io" ]]; then
    echo "ERROR: PWD: $PWD is not project root"
    exit 1
fi

set -x

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

if [[ ${PLATFORM} == "darwin" ]]; then
    N_JOBS=$(sysctl -n hw.ncpu)
else
    N_JOBS=$(grep -c ^processor /proc/cpuinfo)
fi

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

export CC_OPT_FLAGS='-mavx'
export TF_NEED_CUDA=0 # TODO: Verify this is used in GPU custom-op

export PYTHON_BIN_PATH=`which python`

lsb_release -a
ls -la /usr/bin/python*
python --version
python -m pip --version
docker  --version

echo Disabled for now.
exit 0
if [[ $(python -c "import sys;print(sys.version[0])") == "2" ]]; then
  echo Python 2 has been deprecated.
  exit 0
fi

if [[ $(python3 -c 'import sys;print("{}.{}".format(sys.version_info[0],sys.version_info[1]))') == "3.4" ]]; then
  echo Python 3.4 has been deprecated.
  exit 0
fi

export CC_OPT_FLAGS='-mavx'
export TF_NEED_CUDA="1"
export TF_CUDA_VERSION="10.1"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export TF_CUDNN_VERSION="7"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

python3 --version
python3 -m pip --version

python3 -m pip install $(python3 setup.py --install-require)
python3 tools/build/configure.py --cuda

cat .bazelrc

bazel build -s --verbose_failures -c opt -k \
     --jobs=${N_JOBS} \
     --config=linux_ci_gpu \
     //tensorflow_io:python/ops/libtensorflow_io.so

exit $?
