#!/usr/bin/env bash
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

set -e
set -o pipefail

action=$1

if [ "$action" == "start" ]; then

echo ""
echo "Starting the tfio mongodb docker container..."
echo ""
MONGO_IMAGE="mongo"

docker run --rm -d -p 27017-27019:27017-27019 --name tfio-mongodb \
    -e MONGO_INITDB_ROOT_USERNAME=mongoadmin \
    -e MONGO_INITDB_ROOT_PASSWORD=default_password \
    -e MONGO_INITDB_DATABASE=tfiodb \
    ${MONGO_IMAGE}

echo ""
echo "Waiting for mongodb to be up and running..."
echo ""
sleep 60

elif [ "$action" == "stop" ]; then
echo ""
echo "Removing the tfio mongodb container..."
echo ""
docker rm -f tfio-mongodb

else
echo ""
echo "Invalid value: Use 'start' to run the container and 'stop' to remove it."
echo ""
fi
