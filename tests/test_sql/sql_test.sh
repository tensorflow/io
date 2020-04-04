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

set -x
set -e
set -o pipefail

VERSION=12.1
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 start|stop <sql container name>" >&2
  exit 1
fi

if [[ $(uname) == "Darwin" ]]; then
    echo "Not supported on Darwin"
    exit 0
fi

script=$(readlink -f "$0")
base=$(dirname "$script")
echo running from "$base"

action=$1
container=$2
if [ "$action" == "start" ]; then
    sudo apt-get -y -qq update
    sudo apt-get -y -qq install postgresql
    sudo service postgresql start
    sudo -u postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'postgres';"
    sudo -u postgres PGPASSWORD=postgres psql -h localhost -U postgres -f tests/test_sql/run.sql
    echo container $container started successfully
elif [ "$action" == "stop" ]; then
    docker rm -f $container
    echo container $container removed successfully
else
  echo "usage: $0 start|stop <sql container name>" >&2
  exit 1
fi



