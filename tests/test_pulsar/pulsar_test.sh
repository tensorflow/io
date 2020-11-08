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

VERSION=2.6.1
TAR_FILE="apache-pulsar-${VERSION}-bin.tar.gz"

echo "Downloading pulsar ${VERSION}"
if [[ ! -f apache-pulsar-2.6.1-bin.tar.gz ]]; then
  curl -sSOL "https://downloads.apache.org/pulsar/pulsar-2.6.1/${TAR_FILE}"
fi

tar -xzf ${TAR_FILE}
cd "apache-pulsar-${VERSION}"

echo "Disable deleting inactive topics"
sed -i '' "s/brokerDeleteInactiveTopicsFrequencySeconds=.*/brokerDeleteInactiveTopicsFrequencySeconds=86400/" conf/standalone.conf

bin/pulsar-daemon start standalone

echo "-- Wait for Pulsar service to be ready"
until curl http://localhost:8080/metrics > /dev/null 2>&1 ; do sleep 1; done
echo "-- Pulsar service is ready -- Configure permissions"

echo "Creating and populating 'test' topic with sample non-keyed messages"
bin/pulsar-client produce -m "D0,D1,D2,D3,D4,D5" test

echo "Creating and populating 'key-test' topic with sample keyed messages"
bin/pulsar-client produce -m "D0" -k "K0" key-test
bin/pulsar-client produce -m "D1" -k "K1" key-test
bin/pulsar-client produce -m "D2" -k "K0" key-test
bin/pulsar-client produce -m "D3" -k "K1" key-test
bin/pulsar-client produce -m "D4" -k "K0" key-test
bin/pulsar-client produce -m "D5" -k "K1" key-test

echo "Pulsar test setup completed"
exit 0
