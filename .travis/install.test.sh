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
set -e -x

# Install needed repo
DEBIAN_FRONTEND=noninteractive apt-get -y -qq update
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install python python3 ffmpeg > /dev/null

# Install the latest version of pip (needed for google-cloud-pubsub)
curl -sOL https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
python get-pip.py
rm -rf get-pip.py

CPYTHON_VERSION=$(python3 -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')
if [[ ! -z ${TENSORFLOW_INSTALL} ]]; then
  python3 -m pip install -q ${TENSORFLOW_INSTALL}
  python3 -m pip install --no-deps wheelhouse/*-cp${CPYTHON_VERSION}-*.whl
else
  python3 -m pip install wheelhouse/*-cp${CPYTHON_VERSION}-*.whl
fi
python3 -m pip install -q pytest google-cloud-pubsub boto3 pyarrow==0.11.1 pandas==0.19.2
(cd tests && python3 -m pytest --import-mode=append .)

CPYTHON_VERSION=$(python -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')
if [[ ! -z ${TENSORFLOW_INSTALL} ]]; then
  python -m pip install -q ${TENSORFLOW_INSTALL}
  python -m pip install --no-deps wheelhouse/*-cp${CPYTHON_VERSION}-*.whl
else
  python -m pip install wheelhouse/*-cp${CPYTHON_VERSION}-*.whl
fi
python -m pip install -q pytest google-cloud-pubsub boto3 pyarrow==0.11.1 pandas==0.19.2
(cd tests && python -m pytest --import-mode=append .)

if [[ ${TENSORFLOW_INSTALL} == "tf-nightly-2.0-preview" ]]; then
  exit 0
fi

DEBIAN_FRONTEND=noninteractive apt-get -y -qq install software-properties-common apt-transport-https > /dev/null
DEBIAN_FRONTEND=noninteractive apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
DEBIAN_FRONTEND=noninteractive add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran35/"
DEBIAN_FRONTEND=noninteractive apt-get -y -qq update
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install r-base > /dev/null
echo "options(repos = c(CRAN='http://cran.rstudio.com'))" >> ~/.Rprofile
R -e 'install.packages(c("tensorflow"), quiet = TRUE)'
R -e 'install.packages(c("testthat", "devtools"), quiet = TRUE)'
R -e 'install.packages(c("forge"), quiet = TRUE)'
R -e 'library("devtools"); install_github("rstudio/tfdatasets", ref="c6fc59b", quiet = TRUE)'

(cd R-package && R -e 'stopifnot(all(data.frame(devtools::test())$failed == 0L))')
