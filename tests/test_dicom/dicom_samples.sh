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

# change directory to test_dicom folder
cd "${0%/*}"

set -e
set -o pipefail


if [ "$#" -ne 1 ]; then
  echo "Usage: $0 download | extract | clean_{all,dcm}" >&2
  exit 1
fi

if [ "$1" == "download" ]; then
    input="dicom_sample_source.txt"

    while IFS=' ' read -r fname url
    do
    echo "Downloading $fname"
    curl -sL -o $fname $url
    done < "$input"
elif [ "$1" == "extract" ]; then
    input="dicom_sample_source.txt"

    while IFS=' ' read -r fname url
    do
    echo "Extracting $fname"
    gunzip -c $fname > "${fname%.*}.dcm"
    done < "$input"
elif [ "$1" == "clean_all" ]; then
    rm -f *.dcm
    rm -f *.gz
elif [ "$1" == "clean_dcm" ]; then
    rm -f *.dcm
else
  echo "Usage: $0 download | extract | clean_{all,dcm}" >&2
  exit 1
fi
