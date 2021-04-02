# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for GCS file system"""

import os

# Use modular file system plugins from tfio instead of the legacy implementation
# from tensorflow.
os.environ["TF_USE_MODULAR_FILESYSTEM"] = "true"
import sys
import time
import requests
import tensorflow as tf
import tensorflow_io as tfio
import pytest


# GCS emulator setup is in tests/test_gcloud/test_gcs.sh


@pytest.mark.skipif(
    sys.platform in ("win32", "darwin"),
    reason="TODO GCS emulator not setup properly on macOS/Windows yet",
)
def test_read_file():
    """Test case for reading GCS"""

    from google.cloud import storage

    client = storage.Client(
        project="[PROJECT]",
        _http=requests.Session(),
        client_options={"api_endpoint": "http://localhost:9099"},
    )

    body = b"1234567"

    # Setup the GCS bucket and key
    key_name = "TEST"
    bucket_name = "gs{}e".format(int(time.time()))
    bucket = client.create_bucket(bucket_name)

    blob = bucket.blob(key_name)
    blob.upload_from_string(body)

    response = blob.download_as_bytes()
    assert response == body

    os.environ["CLOUD_STORAGE_TESTBENCH_ENDPOINT"] = "http://localhost:9099"

    content = tf.io.read_file("gs://{}/{}".format(bucket_name, key_name))
    assert content == body
