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

if [ "$#" -eq 1 ]; then
  container=$1
  echo pull google/cloud-sdk
  docker pull google/cloud-sdk:236.0.0
  echo pull google/cloud-sdk successfully
  docker run -d --rm --net=host --name=$container-pubsub -v $base:/v -w /v google/cloud-sdk:236.0.0 bash -x -c 'gcloud beta emulators pubsub start'
  echo wait 10 secs until pubsub is up and running
  docker run -d --rm --net=host --name=$container-bigtable -v $base:/v -w /v google/cloud-sdk:236.0.0 bash -x -c 'gcloud beta emulators bigtable start'
  echo wait 10 secs until bigtable is up and running
  sleep 10
  exit 0
fi

curl -sSOL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-236.0.0-darwin-x86_64.tar.gz
tar -xzf google-cloud-sdk-236.0.0-darwin-x86_64.tar.gz
google-cloud-sdk/install.sh -q
google-cloud-sdk/bin/gcloud -q components install beta bigtable cbt
google-cloud-sdk/bin/gcloud -q components install pubsub-emulator
google-cloud-sdk/bin/gcloud -q components update beta
google-cloud-sdk/bin/gcloud -q beta emulators pubsub start &
google-cloud-sdk/bin/gcloud -q beta emulators bigtable start &
exit 0



