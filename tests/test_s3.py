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
"""Tests for S3 file system"""

import os
import sys
import time
import tempfile
import tensorflow as tf
import tensorflow_io as tfio
import pytest


@pytest.mark.skipif(
    sys.platform in ("win32", "darwin"),
    reason="TODO Localstack not setup properly on macOS/Windows yet",
)
def test_read_file():
    """Test case for reading S3"""
    import boto3

    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["AWS_ACCESS_KEY_ID"] = "ACCESS_KEY"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "SECRET_KEY"

    client = boto3.client(
        "s3", region_name="us-east-1", endpoint_url="http://localhost:4566"
    )

    body = b"1234567"

    # Setup the S3 bucket and key
    key_name = "TEST"
    bucket_name = "s3e{}e".format(time.time())

    client.create_bucket(Bucket=bucket_name)
    client.put_object(Bucket=bucket_name, Key=key_name, Body=body)

    response = client.get_object(Bucket=bucket_name, Key=key_name)
    assert response["Body"].read() == body

    os.environ["S3_ENDPOINT"] = "http://localhost:4566"

    content = tf.io.read_file("s3://{}/{}".format(bucket_name, key_name))
    assert content == body
