#  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for S3IOStorage."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import boto3

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position
from tensorflow_io.core.python.ops import core_ops  # pylint: disable=wrong-import-position

def test_s3_read():
  """test_s3_read"""
  os.environ['AWS_ACCESS_KEY_ID'] = 'ACCESS_KEY'
  os.environ['AWS_SECRET_ACCESS_KEY'] = 'SECRET_KEY'
  os.environ['S3_USE_HTTPS'] = '0'
  os.environ['S3_ENDPOINT'] = 'localhost:4572'

  client = boto3.client(
      's3',
      region_name='us-east-1',
      endpoint_url='http://localhost:4572')
  client.create_bucket(Bucket='mybucket')
  for i in range(10):
    body = "D%d" % i
    key = "my/key/%d" % i
    client.put_object(Body=body, Bucket='mybucket', Key=key)

  for i in range(10):
    v = core_ops.storage_read("s3://mybucket/my/key/%d" % i, [], [])
    assert v.numpy().decode() == "D%d" % i

  s3 = tfio.IOStorage.from_s3("s3://mybucket/my/")
  keys = [key.numpy().decode() for key in s3]
  keys.sort()
  assert np.all(keys == ["s3://mybucket/my/key/%d" % i for i in range(10)])

  assert np.all(s3.list().numpy() == [
      b"s3://mybucket/my/key/%d" % i for i in range(10)])

  assert np.all(s3.size().numpy() == [2 for i in range(10)])

if __name__ == "__main__":
  test.main()
