#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

IMAGE_NAME="tfsigio/tfio"
IMAGE_TAG="nightly"
export PYTHON_BIN_PATH=$(which python3)

echo "Build the docker image ..."
docker build -f tools/docker/nightly.Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo "Starting the docker container from image: ${IMAGE_NAME}:${IMAGE_TAG} and validating import ..."
docker run -t --rm ${IMAGE_NAME}:${IMAGE_TAG} python -c "import tensorflow_io as tfio; print(tfio.__version__)"