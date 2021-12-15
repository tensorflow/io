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

# Python version for the base image
ARG PYTHON_VERSION=3.7-slim

FROM python:${PYTHON_VERSION}

# tfio package name and version for pip install
ARG TFIO_PACKAGE=tensorflow-io
ARG TFIO_PACKAGE_VERSION=
ARG TENSORFLOW_VARIANT=tensorflow

RUN pip install ${TFIO_PACKAGE}${TFIO_PACKAGE_VERSION:+==${TFIO_PACKAGE_VERSION}}[${TENSORFLOW_VARIANT}]
