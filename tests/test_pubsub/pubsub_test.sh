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

set -x
set -e
set -o pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 start|stop <pubsub container name>" >&2
  exit 1
fi

if [[ $(uname) == "Darwin" ]]; then
    curl -sSOL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-236.0.0-darwin-x86_64.tar.gz
    tar -xzf google-cloud-sdk-236.0.0-darwin-x86_64.tar.gz
    google-cloud-sdk/install.sh -q
    google-cloud-sdk/bin/gcloud -q components install beta
    google-cloud-sdk/bin/gcloud -q components install pubsub-emulator
    google-cloud-sdk/bin/gcloud -q beta emulators pubsub start &
    exit 0
fi

script=$(readlink -f "$0")
base=$(dirname "$script")
echo running from "$base"

action=$1
container=$2
if [ "$action" == "start" ]; then
    echo pull google/cloud-sdk
    docker pull google/cloud-sdk:236.0.0
    echo pull google/cloud-sdk successfully
    docker run -d --rm --net=host --name=$container -v $base:/v -w /v google/cloud-sdk:236.0.0 bash -x -c 'gcloud beta emulators pubsub start'
    #echo wait 10 secs until pubsub is up and running
    #sleep 10
elif [ "$action" == "stop" ]; then
    docker rm -f $container
    echo container $container removed successfully
else
  echo "usage: $0 start|stop <pubsub container name>" >&2
  exit 1
fi



