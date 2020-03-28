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

VERSION=5.4.1

if [[ "$(uname)" == "Darwin" ]]; then
    curl -sSOL http://packages.confluent.io/archive/5.4/confluent-community-5.4.1-2.12.tar.gz
    tar -xzf confluent-community-5.4.1-2.12.tar.gz
    (cd confluent-$VERSION/ && sudo bin/zookeeper-server-start -daemon etc/kafka/zookeeper.properties)
    echo Wait 10 secs until zookeeper is up and running
    sleep 10
    (cd confluent-$VERSION/ && sudo bin/kafka-server-start -daemon etc/kafka/server.properties)
    echo Wait 10 secs until kafka is up and running
    sleep 10
    (cd confluent-$VERSION/ && sudo bin/schema-registry-start -daemon etc/schema-registry/schema-registry.properties)
    echo -e "D0\nD1\nD2\nD3\nD4\nD5\nD6\nD7\nD8\nD9" > confluent-$VERSION/test
    echo -e "K0:D0\nK1:D1\nK0:D2\nK1:D3\nK0:D4\nK1:D5\nK0:D6\nK1:D7\nK0:D8\nK1:D9" > confluent-$VERSION/key-test
    echo Wait 15 secs until all is up and running
    sleep 15
    sudo confluent-$VERSION/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test
    sudo confluent-$VERSION/bin/kafka-console-producer --topic test --broker-list 127.0.0.1:9092 < confluent-$VERSION/test
    sudo confluent-$VERSION/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic key-test
    sudo confluent-$VERSION/bin/kafka-console-producer --topic key-test --property "parse.key=true" --property "key.separator=:" --broker-list 127.0.0.1:9092 < confluent-$VERSION/key-test
    sudo confluent-$VERSION/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic avro-test
    echo -e "{\"f1\":\"value1\",\"f2\":1,\"f3\":null}\n{\"f1\":\"value2\",\"f2\":2,\"f3\":{\"string\":\"2\"}}\n{\"f1\":\"value3\",\"f2\":3,\"f3\":null}" > confluent-$VERSION/avro-test
    sudo confluent-$VERSION/bin/kafka-avro-console-producer --broker-list localhost:9092 --topic avro-test --property value.schema="{\"type\":\"record\",\"name\":\"myrecord\",\"fields\":[{\"name\":\"f1\",\"type\":\"string\"},{\"name\":\"f2\",\"type\":\"long\"},{\"name\":\"f3\",\"type\":[\"null\",\"string\"],\"default\":null}]}" < confluent-$VERSION/avro-test
    echo Everything started
    exit 0
fi
action=$1
container=$2
if [ "$action" == "start" ]; then
    docker pull confluentinc/cp-zookeeper:$VERSION
    docker pull confluentinc/cp-kafka:$VERSION
    docker pull confluentinc/cp-schema-registry:$VERSION
    docker run -d \
      --net=host \
      --name=$container-zookeeper \
      -e ZOOKEEPER_CLIENT_PORT=2181 \
      -e ZOOKEEPER_TICK_TIME=2000 \
      -e ZOOKEEPER_SYNC_LIMIT=2 \
      confluentinc/cp-zookeeper:$VERSION
    echo Container $container-zookeeper started successfully
    docker run -d \
      --net=host \
      --name=$container-kafka \
      -e KAFKA_ZOOKEEPER_CONNECT=localhost:2181 \
      -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
      -e KAFKA_BROKER_ID=2 \
      -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
      confluentinc/cp-kafka:$VERSION
    echo Container $container-kafka started successfully
    echo Wait 10 secs until kafka is up and running
    docker run -d \
      --net=host \
      --name=$container-schema-registry \
      -e SCHEMA_REGISTRY_KAFKASTORE_CONNECTION_URL=localhost:2181 \
      -e SCHEMA_REGISTRY_HOST_NAME=localhost \
      -e SCHEMA_REGISTRY_LISTENERS=http://localhost:8081 \
      -e SCHEMA_REGISTRY_DEBUG=true \
      confluentinc/cp-schema-registry:$VERSION
    echo Wait 15 secs until all is up and running
    sleep 15
    echo Create test topic
    docker exec -i $container-kafka bash -c '/usr/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test'
    echo Create test message
    docker exec -i $container-kafka bash -c 'echo -e "D0\nD1\nD2\nD3\nD4\nD5\nD6\nD7\nD8\nD9" > /test'
    docker exec -i $container-kafka bash -c 'echo -e "K0:D0\nK1:D1\nK0:D2\nK1:D3\nK0:D4\nK1:D5\nK0:D6\nK1:D7\nK0:D8\nK1:D9" > /key-test'
    echo Produce test message
    docker exec -i $container-kafka bash -c '/usr/bin/kafka-console-producer --topic test --broker-list 127.0.0.1:9092 < /test'
    docker exec -i $container-kafka bash -c '/usr/bin/kafka-console-producer --topic key-test --property "parse.key=true" --property "key.separator=:" --broker-list 127.0.0.1:9092 < /key-test'
    echo Create avro test topic
    docker exec -i $container-kafka bash -c '/usr/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic avro-test'
    echo Create avro test message
    docker exec -i $container-schema-registry bash -c 'echo -e "{\"f1\":\"value1\",\"f2\":1,\"f3\":null}\n{\"f1\":\"value2\",\"f2\":2,\"f3\":{\"string\":\"2\"}}\n{\"f1\":\"value3\",\"f2\":3,\"f3\":null}" > /avro-test'
    echo Produce avro test message
    docker exec -i $container-schema-registry bash -c "/usr/bin/kafka-avro-console-producer --broker-list localhost:9092 --topic avro-test --property value.schema='{\"type\":\"record\",\"name\":\"myrecord\",\"fields\":[{\"name\":\"f1\",\"type\":\"string\"},{\"name\":\"f2\",\"type\":\"long\"},{\"name\":\"f3\",\"type\":[\"null\",\"string\"],\"default\":null}]}' </avro-test"
    echo Container $container started successfully
elif [ "$action" == "stop" ]; then
    docker rm -f $container-zookeeper
    docker rm -f $container-kafka
    docker rm -f $container-schema-registry
    echo Container $container removed successfully
else
  echo "Usage: $0 start|stop <kafka container name>" >&2
  exit 1
fi



