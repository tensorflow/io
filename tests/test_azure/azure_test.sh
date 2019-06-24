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

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 start|stop <azure container name>" >&2
  exit 1
fi

action=$1
container=$2
if [ "$action" == "start" ]; then
    echo pull arafato/azurite
    docker pull arafato/azurite
    echo pull arafato/azurite successfully
    docker run -d --rm -p 10000:10000 --name=$container arafato/azurite
    echo Container $container started successfully
elif [ "$action" == "stop" ]; then
    docker rm -f $container
    echo Container $container removed successfully
else
  echo "Usage: $0 start|stop <azure container name>" >&2
  exit 1
fi
