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

export TENSORFLOW_INSTALL="$(python setup.py --package-version)"
if [[ "$#" -gt 0 ]]; then
    export TENSORFLOW_INSTALL="${1}"
fi

export BAZEL_VERSION=0.29.0 BAZEL_OS=$(uname | tr '[:upper:]' '[:lower:]')
curl -sSOL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
bash -e bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh
bazel version

if [[ $(uname) == "Linux" ]]; then
  curl -sSOL https://bootstrap.pypa.io/get-pip.py
  python get-pip.py -q
  python -m pip --version
else
# upgrading pip version on MacOS due to 
# https://github.com/googleapis/google-cloud-python/issues/2990
  python -m pip install --upgrade pip
  python -m pip install --upgrade setuptools
  python -m pip --version
fi

python -m pip install -q ${TENSORFLOW_INSTALL}
python -m pip install gast==0.2.2

python third_party/tf/configure.py

cat .bazelrc

args=()
if [[ $(uname) == "Darwin" ]]; then
  # The -DGRPC_BAZEL_BUILD is needed because
  # gRPC does not compile on macOS unless it is set.
  args+=(--copt=-DGRPC_BAZEL_BUILD)
else
  # Build with manulinux2010
  args+=(--crosstool_top=//third_party/toolchain:toolchain)
fi
if [[ "${BAZEL_CACHE}" != "" ]]; then
  args+=(--disk_cache=${BAZEL_CACHE})
fi

bazel build \
  "${args[@]}" \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  -- //tensorflow_io/arrow:arrow_ops \
     //tensorflow_io/audio:audio_ops \
     //tensorflow_io/avro:avro_ops \
     //tensorflow_io/azure:azfs_ops \
     //tensorflow_io/bigquery:bigquery_ops \
     //tensorflow_io/core:core_ops \
     //tensorflow_io/dicom:dicom_ops \
     //tensorflow_io/gcs:gcs_config_ops
     //tensorflow_io/genome:genome_ops \
     //tensorflow_io/grpc:grpc_ops \
     //tensorflow_io/hdf5:hdf5_ops \
     //tensorflow_io/ignite:ignite_ops \
     //tensorflow_io/image:image_ops \
     //tensorflow_io/json:json_ops \
     //tensorflow_io/kafka:kafka_ops \
     //tensorflow_io/kinesis:kinesis_ops \
     //tensorflow_io/libsvm:libsvm_ops

rm -rf build && mkdir -p build

cp -r bazel-bin/tensorflow_io  build/tensorflow_io

exit 0
