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

# Examples: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/data/experimental/kernel_tests/stats_dataset_test_base.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tempfile

from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework.errors import OpError, OutOfRangeError
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.util import compat
from tensorflow_io.avro.python.ops import avro_ops
from tensorflow_io.avro.python.utils.avro_serialization import \
  AvroRecordsToFile


class AvroDatasetTestBase(test_base.DatasetTestBase):

  def test_primitive_types(self):
    schema = """{
             "type": "record",
             "name": "dataTypes",
             "fields": [
                 {  
                    "name":"string_value",
                    "type":"string"
                 },
                 {  
                    "name":"double_value",
                    "type":"double"
                 },
                 {  
                    "name":"float_value",
                    "type":"float"
                 },
                 {  
                    "name":"long_value",
                    "type":"long"
                 },
                 {  
                    "name":"int_value",
                    "type":"int"
                 },
                 {  
                    "name":"boolean_value",
                    "type":"boolean"
                 }
             ]}"""
    record_data = [
      {
        "string_value": "",
        "double_value": 0.0,
        "float_value": 0.0,
        "long_value": 0,
        "int_value": 0,
        "boolean_value": False,
      },
      {
        "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
        "double_value": 1.7976931348623157e+308,
        "float_value": 3.40282306074e+38,
        "long_value": 9223372036854775807,
        "int_value": 2147483648 - 1,
        "boolean_value": True,
      },
      {
        "string_value": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
        "double_value": -1.7976931348623157e+308,
        "float_value": -3.40282306074e+38,
        "long_value": -9223372036854775807 - 1,
        "int_value": -2147483648,
        "boolean_value": False,
      }
    ]
    columns = ["string_value", "double_value", "float_value",
               "long_value", "int_value", "boolean_value"]
    dtypes = (tf_types.string, tf_types.float64, tf_types.float32,
              tf_types.int64, tf_types.int32, tf_types.bool)
    expected_tensors = [
        (
        np.asarray([
          compat.as_bytes(""),
          compat.as_bytes("SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"),
          compat.as_bytes("ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789")
        ]),
        np.asarray([
          0.0,
          1.7976931348623157e+308,
          -1.7976931348623157e+308
        ]),
        np.asarray([
          0.0,
          3.40282306074e+38,
          -3.40282306074e+38
        ]),
        np.asarray([
          0,
          9223372036854775807,
          -9223372036854775807-1
        ]),
        np.asarray([
          0,
          2147483648-1,
          -2147483648
        ]),
        np.asarray([
          False,
          True,
          False
        ])
      )
    ]
    self._test_pass_dataset(schema=schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            columns=columns,
                            dtypes=dtypes,
                            batch_size=3)

  @staticmethod
  def _setup_files(writer_schema, records):

    # Write test records into temporary output directory
    filename = os.path.join(tempfile.mkdtemp(), "test.avro")
    writer = AvroRecordsToFile(filename=filename,
                               writer_schema=writer_schema)
    writer.write_records(records)

    return [filename]

  def _assert_same_tensors(self, expected_tensors, actual_tensors):

    assert len(expected_tensors) == len(actual_tensors), \
      "Expected {} pairs but got {} pairs".format(
          len(expected_tensors), len(actual_tensors))

    for i_tensor in range(len(expected_tensors)):
      actual_tensor = actual_tensors[i_tensor]
      expected_tensor = expected_tensors[i_tensor]

      def assert_same_array(expected_array, actual_array):
        if np.issubdtype(actual_array.dtype, np.number):
          self.assertAllClose(expected_array, actual_array)
        else:
          self.assertAllEqual(expected_array, actual_array)

      assert_same_array(expected_tensor, actual_tensor)

  def _verify_output(self, expected_tensors, actual_dataset):

    # Turn off any parallelism and random for testing
    config = config_pb2.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    with self.test_session(config=config) as sess:

      iterator = actual_dataset.make_initializable_iterator()
      next_element = iterator.get_next()
      sess.run(iterator.initializer)

      for tensors in expected_tensors:
        self._assert_same_tensors(expected_tensors=tensors,
                                  actual_tensors=sess.run(next_element))

      with self.assertRaises(OutOfRangeError):
        sess.run(next_element)

  def _test_pass_dataset(self, schema, record_data, expected_tensors,
      columns, dtypes, batch_size):
    filenames = AvroDatasetTestBase._setup_files(writer_schema=schema,
                                                 records=record_data)

    actual_dataset = avro_ops.AvroDataset(
        filenames=filenames, columns=columns, schema=schema, dtypes=dtypes,
        batch=batch_size)

    self._verify_output(expected_tensors=expected_tensors,
                        actual_dataset=actual_dataset)


if __name__ == "__main__":
  test.main()
