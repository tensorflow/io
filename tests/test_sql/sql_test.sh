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
if [ "$#" -eq 1 ]; then
  script=$(readlink -f "$0")
  base=$(dirname "$script")
  echo running from "$base"

  container=$1
  echo pull postgres:$VERSION
  docker pull postgres:$VERSION
  echo pull postgres:$VERSION successfully
  docker run -d --rm --net=host --name=$container -v $base:/v -w /v postgres:$VERSION
  #echo wait 10 secs until container is up and running
  sleep 10
  docker exec -i $container bash -c "psql -h localhost -U postgres -f run.sql"
  echo container $container started successfully

  exit 0
fi

if [[ $(uname) == "Darwin" ]]; then
    echo "Not supported on Darwin"
    exit 0
fi

script=$(readlink -f "$0")
base=$(dirname "$script")
echo running from "$base"

sudo apt-get -y -qq update
sudo apt-get -y -qq install postgresql
sudo service postgresql start
sudo -u postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres PGPASSWORD=postgres psql -h localhost -U postgres -f tests/test_sql/run.sql
echo container $container started successfully
exit 0
