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
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev \
    software-properties-common apt-transport-https \
    python-pip python3-pip \
    ffmpeg > /dev/null
DEBIAN_FRONTEND=noninteractive apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
DEBIAN_FRONTEND=noninteractive add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran35/"
DEBIAN_FRONTEND=noninteractive apt-get -y -qq update
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install r-base > /dev/null
echo "options(repos = c(CRAN='http://cran.rstudio.com'))" >> ~/.Rprofile
R -e 'install.packages(c("tensorflow", "tfdatasets"), quiet = TRUE)'
R -e 'install.packages(c("testthat", "devtools"), quiet = TRUE)'
R -e 'install.packages(c("forge"), quiet = TRUE)'

V=$(python3 -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')
python3 -m pip install dist/*${V}*.whl
python3 -m pip install -q pytest boto3 pyarrow==0.11.1 pandas==0.19.2
(cd tests && python3 -m pytest .)

V=$(python -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')
python -m pip install dist/*${V}*.whl
python -m pip install -q pytest boto3 pyarrow==0.11.1 pandas==0.19.2
(cd tests && python -m pytest .)

(cd R-package && R -e 'stopifnot(all(data.frame(devtools::test())$failed == 0L))')
