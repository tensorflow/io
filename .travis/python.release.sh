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

# Release:

if [[ "$1" == "--nightly" ]]; then
  NIGHTLY_NUMBER=$2
  shift
  shift
fi

if [[ $(uname) == "Darwin" ]]; then
  bash -x -e .travis/bazel.build.sh $@

  python setup.py --data build -q bdist_wheel

  if [[ "$NIGHTLY_NUMBER" != "" ]]; then
    python setup.py --data build -q bdist_wheel --nightly $NIGHTLY_NUMBER
  fi

  ls dist/*
  for f in dist/*.whl; do
    delocate-wheel -w wheelhouse  $f
  done
  ls wheelhouse/*
else
  docker run -i --rm -v $PWD:/v -w /v --net=host -e BAZEL_CACHE=${BAZEL_CACHE} -e BAZEL_CPU_OPTIMIZATION=${BAZEL_CPU_OPTIMIZATION} gcr.io/tensorflow-testing/nosla-ubuntu16.04-manylinux2010@sha256:3a9b4820021801b1fa7d0592c1738483ac7abc209fc6ee8c9ef06cf2eab2d170 /v/.travis/bazel.build.sh $@
  sudo chown -R $(id -nu):$(id -ng) .

  for entry in 2.7 3.5 3.6 3.7; do
    docker run -i --rm --user $(id -u):$(id -g) -v /etc/password:/etc/password -v $PWD:/v -w /v --net=host python:${entry}-slim python setup.py --data build -q bdist_wheel
    if [[ "$NIGHTLY_NUMBER" != "" ]]; then
      docker run -i --rm --user $(id -u):$(id -g) -v /etc/password:/etc/password -v $PWD:/v -w /v --net=host python:${entry}-slim python setup.py --data build -q bdist_wheel --nightly $NIGHTLY_NUMBER
    fi
  done

  ls dist/*
  for f in dist/*.whl; do
    docker run -i --rm -v $PWD:/v -w /v --net=host quay.io/pypa/manylinux2010_x86_64 bash -x -e /v/third_party/tf/auditwheel repair --plat manylinux2010_x86_64 $f
  done
  sudo chown -R $(id -nu):$(id -ng) .
  ls wheelhouse/*
fi
