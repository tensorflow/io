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

IGNITE_VERSION=2.6.0
SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

if [[ "$(uname)" == "Darwin" ]]; then
    set -e
    curl -sSOL https://archive.apache.org/dist/ignite/${IGNITE_VERSION}/apache-ignite-fabric-${IGNITE_VERSION}-bin.zip
    unzip -qq apache-ignite-fabric-${IGNITE_VERSION}-bin.zip
    apache-ignite-fabric-${IGNITE_VERSION}-bin/bin/ignite.sh tests/test_ignite/config/ignite-config-plain.xml &
    sleep 10 # Wait Apache Ignite to be started
    apache-ignite-fabric-${IGNITE_VERSION}-bin/bin/sqlline.sh -u "jdbc:ignite:thin://127.0.0.1/" --run=tests/test_ignite/sql/init.sql
    apache-ignite-fabric-${IGNITE_VERSION}-bin/bin/ignite.sh tests/test_ignite/config/ignite-config-igfs.xml &
    sleep 10 # Wait Apache Ignite to be started
    exit 0
fi

# Start Apache Ignite with plain client listener.
docker run -itd --name ignite-plain -p 10800:10800 \
-v ${SCRIPT_PATH}:/data apacheignite/ignite:${IGNITE_VERSION} /data/bin/start-plain.sh

# Start Apache Ignite with IGFS.
docker run -itd --name ignite-igfs -p 10500:10500 \
-v ${SCRIPT_PATH}:/data apacheignite/ignite:${IGNITE_VERSION} /data/bin/start-igfs.sh

# Start GridGain CE with GGFS.
docker run -itd --name gridgain-ce-ml -p 10801:10801 \
-v ${SCRIPT_PATH}:/data dmitrievanthony/gridgain-ce-ml /data/bin/start-ggfs.sh

# Wait Apache Ignite to be started
sleep 10
