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
set -o pipefail

LOCALSTACK_VERSION=0.12.10
docker pull localstack/localstack:$LOCALSTACK_VERSION
docker run -d --rm --net=host --name=tensorflow-io-aws localstack/localstack:$LOCALSTACK_VERSION
echo "Waiting for 10 secs until localstack is up and running"
sleep 10
echo "Localstack up"
exit 0
