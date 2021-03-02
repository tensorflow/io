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

HADOOP_VERSION=2.7.0
docker pull sequenceiq/hadoop-docker:$HADOOP_VERSION
docker run -d --rm -p 9000:9000 --name=tensorflow-io-hdfs sequenceiq/hadoop-docker:$HADOOP_VERSION
echo "Waiting for 30 secs until hadoop is up and running"
sleep 30
docker logs tensorflow-io-hdfs
echo "Hadoop up"
exit 0
