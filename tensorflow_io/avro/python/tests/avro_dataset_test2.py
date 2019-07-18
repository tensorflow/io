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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat
from tensorflow.python.framework import sparse_tensor

from tensorflow_io.avro.python.tests import avro_dataset_test_base as \
  avro_test_base


class AvroDatasetTest(avro_test_base.AvroDatasetTestBase):

  def test_batching(self):
    writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
    record_data = [
      {"int_value": 0},
      {"int_value": 1},
      {"int_value": 2}
    ]
    features = {
      "int_value": parsing_ops.FixedLenFeature([], tf_types.int32)
    }
    expected_tensors = [
      {"int_value": np.asarray([0, 1])},
      {"int_value": np.asarray([2])}
    ]
    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
                            batch_size=2, num_epochs=1, reader_schema=writer_schema)

if __name__ == "__main__":
  test.main()
