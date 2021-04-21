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

VERSION=2.6.1
TAR_FILE="apache-pulsar-${VERSION}-bin.tar.gz"

echo "Downloading pulsar ${VERSION}"
if [[ ! -f ${TAR_FILE} ]]; then
  curl -sSOL "https://archive.apache.org/dist/pulsar/pulsar-${VERSION}/${TAR_FILE}"
fi

tar -xzf ${TAR_FILE}
cd "apache-pulsar-${VERSION}"

echo "Disable deleting inactive topics"
sed -i.bak 's/zookeeperServers=.*/zookeeperServers=localhost:2182/' conf/standalone.conf
sed -i.bak "s/brokerDeleteInactiveTopicsFrequencySeconds=.*/brokerDeleteInactiveTopicsFrequencySeconds=86400/" conf/standalone.conf
sed -i.bak 's/advertisedAddress=.*/advertisedAddress=127.0.0.1/' conf/standalone.conf
sed -i.bak 's/bindAddress=.*/bindAddress=127.0.0.1/' conf/standalone.conf

bin/pulsar-daemon start standalone

echo "Waiting for Pulsar service ready or 30 seconds passed"
for i in {1..30}; do
  RESPONSE=$(curl --write-out '%{http_code}' --silent -o /dev/null -L http://127.0.0.1:8080/admin/v2/persistent/public/default) || true
  if [[ $RESPONSE == 200 ]]; then
      echo "[$i] Access namespace public/default successfully"
      break
  fi
  echo "[$i] Access namespace public/default failed: $RESPONSE, sleep for 1 second"
  sleep 1
done
echo "Sleep for 5 seconds more to avoid flaky test"
sleep 5

echo "Creating and populating 'test' topic with sample non-keyed messages"
bin/pulsar-client --url pulsar://127.0.0.1:6650 produce -m "D0,D1,D2,D3,D4,D5" test

echo "Creating and populating 'key-test' topic with sample keyed messages"
bin/pulsar-client --url pulsar://127.0.0.1:6650 produce -m "D0" -k "K0" key-test
bin/pulsar-client --url pulsar://127.0.0.1:6650 produce -m "D1" -k "K1" key-test
bin/pulsar-client --url pulsar://127.0.0.1:6650 produce -m "D2" -k "K0" key-test
bin/pulsar-client --url pulsar://127.0.0.1:6650 produce -m "D3" -k "K1" key-test
bin/pulsar-client --url pulsar://127.0.0.1:6650 produce -m "D4" -k "K0" key-test
bin/pulsar-client --url pulsar://127.0.0.1:6650 produce -m "D5" -k "K1" key-test

echo "Pulsar test setup completed"
exit 0
