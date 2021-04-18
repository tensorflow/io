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

VERSION=6.1.1

echo "Downloading the confluent packages"
curl -sSOL http://packages.confluent.io/archive/6.1/confluent-community-6.1.1.tar.gz
tar -xzf confluent-community-6.1.1.tar.gz
echo 'advertised.listeners=PLAINTEXT://127.0.0.1:9092' >> confluent-$VERSION/etc/kafka/server.properties

(cd confluent-$VERSION/ && sudo bin/zookeeper-server-start -daemon etc/kafka/zookeeper.properties)
echo "Waiting for 10 secs until zookeeper is up and running"
sleep 10

(cd confluent-$VERSION/ && sudo bin/kafka-server-start -daemon etc/kafka/server.properties)
echo "Waiting for 10 secs until kafka is up and running"
sleep 10

(cd confluent-$VERSION/ && sudo bin/schema-registry-start -daemon etc/schema-registry/schema-registry.properties)
echo -e "D0\nD1\nD2\nD3\nD4\nD5\nD6\nD7\nD8\nD9" > confluent-$VERSION/test
echo -e "K0:D0\nK1:D1\nK0:D2\nK1:D3\nK0:D4\nK1:D5\nK0:D6\nK1:D7\nK0:D8\nK1:D9" > confluent-$VERSION/key-test
echo -e "K0:D0\nK1:D1\nK0:D2\nK1:D3\nK0:D4\nK1:D5\nK0:D6\nK1:D7\nK0:D8\nK1:D9" > confluent-$VERSION/key-partition-test
echo -e "0:0\n1:1\n0:2\n1:3\n0:4\n1:5\n0:6\n1:7\n0:8\n1:9" > confluent-$VERSION/mini-batch-test
echo -e "D0\nD1\nD2\nD3\nD4\nD5\nD6\nD7\nD8\nD9" > confluent-$VERSION/offset-test
echo "Waiting for 30 secs until schema registry is ready and other services are up and running"
sleep 30

echo "Creating and populating 'test' topic with sample non-keyed messages"
sudo confluent-$VERSION/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test
sudo confluent-$VERSION/bin/kafka-console-producer --topic test --broker-list 127.0.0.1:9092 < confluent-$VERSION/test

echo "Creating and populating 'key-test' topic with sample keyed messages"
sudo confluent-$VERSION/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic key-test
sudo confluent-$VERSION/bin/kafka-console-producer --topic key-test --property "parse.key=true" --property "key.separator=:" --broker-list 127.0.0.1:9092 < confluent-$VERSION/key-test

echo "Creating and populating 'key-partition-test' multi-partition topic with sample keyed messages"
sudo confluent-$VERSION/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 2 --topic key-partition-test
sudo confluent-$VERSION/bin/kafka-console-producer --topic key-partition-test --property "parse.key=true" --property "key.separator=:" --broker-list 127.0.0.1:9092 < confluent-$VERSION/key-partition-test

echo "Creating and populating 'mini-batch-test' multi-partition topic with sample keyed messages"
sudo confluent-$VERSION/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 2 --topic mini-batch-test
sudo confluent-$VERSION/bin/kafka-console-producer --topic mini-batch-test --property "parse.key=true" --property "key.separator=:" --broker-list 127.0.0.1:9092 < confluent-$VERSION/mini-batch-test

echo "Creating and populating 'offset-test' topic with sample non-keyed messages"
sudo confluent-$VERSION/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic offset-test
sudo confluent-$VERSION/bin/kafka-console-producer --topic offset-test --broker-list 127.0.0.1:9092 < confluent-$VERSION/offset-test


echo "Creating and populating 'avro-test' topic with sample messages."
sudo confluent-$VERSION/bin/kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic avro-test
echo -e "{\"f1\":\"value1\",\"f2\":1,\"f3\":null}\n{\"f1\":\"value2\",\"f2\":2,\"f3\":{\"string\":\"2\"}}\n{\"f1\":\"value3\",\"f2\":3,\"f3\":null}" > confluent-$VERSION/avro-test
sudo confluent-$VERSION/bin/kafka-avro-console-producer --broker-list localhost:9092 --topic avro-test --property value.schema="{\"type\":\"record\",\"name\":\"myrecord\",\"fields\":[{\"name\":\"f1\",\"type\":\"string\"},{\"name\":\"f2\",\"type\":\"long\"},{\"name\":\"f3\",\"type\":[\"null\",\"string\"],\"default\":null}]}" < confluent-$VERSION/avro-test

echo "Kafka test setup completed."
exit 0
