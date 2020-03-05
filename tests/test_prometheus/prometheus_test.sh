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

if [[ "$(uname)" == "Darwin" ]]; then
    exit 0
fi

action=$1

if [ "$action" == "start" ]; then
cat <<EOF >.coredns
.:1053 {
  whoami 
  prometheus
}
EOF

docker run -d --rm --name=tensorflow-io-coredns --net=host -v $PWD/.coredns:/Corefile coredns/coredns

cat <<EOF >.prometheus
global:
  scrape_interval:     1s
  evaluation_interval: 1s
alerting:
  alertmanagers:
  - static_configs:
    - targets:
rule_files:
scrape_configs:
- job_name: 'prometheus'
  static_configs:
  - targets: ['localhost:9090']
- job_name: "coredns"
  static_configs:
  - targets: ['localhost:9153']
EOF

docker run -d --rm --name=tensorflow-io-prometheus --net=host -v $PWD/.prometheus:/etc/prometheus/prometheus.yml prom/prometheus

# wait for coredns and prometheus up
sleep 5

dig @localhost -p 1053 www.google.com
dig @localhost -p 1053 www.google.com
dig @localhost -p 1053 www.google.com
dig @localhost -p 1053 www.google.com
dig @localhost -p 1053 www.google.com
dig @localhost -p 1053 www.google.com

else
docker rm -f tensorflow-io-coredns
docker rm -f tensorflow-io-prometheus
fi
