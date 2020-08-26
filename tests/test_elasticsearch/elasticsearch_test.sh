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

echo "Preparing the environment variables file..."
cat >> .env-vars << "EOF"
cluster.name=tfio-es-cluster
bootstrap.memory_lock=true
discovery.type=single-node
EOF

echo "Starting the tfio elasticsearch docker container..."
ELASTICSEARCH_IMAGE="docker.elastic.co/elasticsearch/elasticsearch:7.4.0"

docker run -d --rm --name=tfio-elasticsearch \
-p 9200:9200 \
--env-file ./.env-vars \
--ulimit memlock=-1:-1 \
${ELASTICSEARCH_IMAGE}

echo "Waiting for the elasticsearch cluster to be up and running..."
sleep 20

echo "Checking the base REST-API endpoint..."
curl localhost:9200/

echo "Checking the healthcheck REST-API endpoint..."
curl localhost:9200/_cluster/health

echo "Clean up..."
rm -rf ./.env-vars

elif [ "$action" == "stop" ]; then
echo "Removing the tfio elasticsearch container..."
docker rm -f tfio-elasticsearch

else
echo "Invalid value: Use 'start' to run the container and 'stop' to remove it."
fi
