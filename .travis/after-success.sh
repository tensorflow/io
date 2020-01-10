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

if [[ ( ${TRAVIS_BRANCH} == "master" ) && ( ${TRAVIS_EVENT_TYPE} != "pull_request" ) ]]; then

  # twine upload wheelhouse/tensorflow_io_nightly-*.whl

  for entry in wheelhouse/tensorflow_io-*.whl ; do
    if [[ $(uname) == "Darwin" ]]; then
      shasum -a 256 $entry
    else
      sha256sum $entry
    fi

    STATUS=$(curl --write-out %{http_code} --silent --output /dev/null -X POST https://content.dropboxapi.com/2/files/upload --header "Authorization: Bearer ${DROPBOX_ACCESS_TOKEN}" --header "Dropbox-API-Arg: {\"path\": \"/release/${TRAVIS_BUILD_NUMBER}/$entry\",\"mode\": \"add\",\"autorename\": true,\"mute\": false,\"strict_conflict\": false}" --header "Content-Type: application/octet-stream" --data-binary @$entry)

    if [[ "$STATUS" -ne 200 ]] ; then
      echo "Upload to Dropbox: $STATUS"
      exit 1
    fi
  done
fi
