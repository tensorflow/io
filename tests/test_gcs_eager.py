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

    # Setup the S3 bucket and key
    key_name = "TEST"
    bucket_name = "s3e{}e".format(int(time.time()))

    bucket = client.create_bucket(bucket_name)
    print("Project number: {}".format(bucket.project_number))

    blob = bucket.blob(key_name)
    blob.upload_from_string(body)

    response = blob.download_as_string()
    print("RESPONSE: ", response)
    assert response == body

    # content = tf.io.read_file("gs://{}/{}".format(bucket_name, key_name))
    # assert content == body
