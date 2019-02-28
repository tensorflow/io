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
  echo "Usage: $0 start|stop <kafka container name>" >&2
  exit 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
    curl -sSOL https://archive.apache.org/dist/kafka/0.10.1.0/kafka_2.11-0.10.1.0.tgz
    tar -xzf kafka_2.11-0.10.1.0.tgz
    (cd kafka_2.11-0.10.1.0/ && bin/zookeeper-server-start.sh -daemon config/zookeeper.properties)
    (cd kafka_2.11-0.10.1.0/ && bin/kafka-server-start.sh -daemon config/server.properties)
    echo -e "D0\nD1\nD2\nD3\nD4\nD5\nD6\nD7\nD8\nD9" > kafka_2.11-0.10.1.0/test
    echo Wait 15 secs until kafka is up and running
    sleep 5
    kafka_2.11-0.10.1.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test
    kafka_2.11-0.10.1.0/bin/kafka-console-producer.sh --topic test --broker-list 127.0.0.1:9092 < kafka_2.11-0.10.1.0/test
    exit 0
fi
action=$1
container=$2
if [ "$action" == "start" ]; then
    echo pull spotify/kafka
    docker pull spotify/kafka
    echo pull spotify/kafka successfully
    docker run -d --rm --net=host --name=$container spotify/kafka
    echo Wait 5 secs until kafka is up and running
    sleep 5
    echo Create test topic
    docker exec $container bash -c '/opt/kafka_2.11-0.10.1.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test'
    echo Create test message
    docker exec $container bash -c 'echo -e "D0\nD1\nD2\nD3\nD4\nD5\nD6\nD7\nD8\nD9" > /test'
    echo Produce test message
    docker exec $container bash -c '/opt/kafka_2.11-0.10.1.0/bin/kafka-console-producer.sh --topic test --broker-list 127.0.0.1:9092 < /test'
    echo Container $container started successfully
elif [ "$action" == "stop" ]; then
    docker rm -f $container
    echo Container $container removed successfully
else
  echo "Usage: $0 start|stop <kafka container name>" >&2
  exit 1
fi



