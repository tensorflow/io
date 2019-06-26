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
if [[ $(uname) == "Darwin" ]]; then
    ls -la /usr/local/bin/n*
    curl "https://nodejs.org/dist/latest/node-${VERSION:-$(wget -qO- https://nodejs.org/dist/latest/ | sed -nE 's|.*>node-(.*)\.pkg</a>.*|\1|p')}.pkg" > "$HOME/Downloads/node-latest.pkg" && sudo installer -store -pkg "$HOME/Downloads/node-latest.pkg" -target "/"
    /usr/local/bin/node --version
    /usr/local/bin/npm --version
    sudo rm -rf /Users/travis/.npm
    /usr/local/bin/npm install -g azurite
    azurite -l /tmp --blobHost 0.0.0.0
    exit 0
fi

action=$1
container=$2
if [ "$action" == "start" ]; then
    echo pull arafato/azurite
    docker pull arafato/azurite
    echo pull arafato/azurite successfully
    docker run -d --rm --net=host --name=$container arafato/azurite
    echo Container $container started successfully
elif [ "$action" == "stop" ]; then
    docker rm -f $container
    echo Container $container removed successfully
else
  echo "Usage: $0 start|stop <azure container name>" >&2
  exit 1
fi
