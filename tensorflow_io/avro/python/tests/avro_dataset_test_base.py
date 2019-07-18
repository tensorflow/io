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

import logging
import numpy as np
import os
import six
import tempfile

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import test_util, sparse_tensor
from tensorflow.python.framework.errors import OpError, OutOfRangeError
from tensorflow.python.data.kernel_tests import test_base
from tensorflow_io.avro.python.ops import avro_dataset
from tensorflow_io.avro.python.utils.avro_serialization import \
  AvroRecordsToFile


@test_util.run_all_in_graph_and_eager_modes
class AvroDatasetTestBase(test_base.DatasetTestBase):

  @staticmethod
  def _setup_files(writer_schema, records):

    # Write test records into temporary output directory
    filename = os.path.join(tempfile.mkdtemp(), "test.avro")
    writer = AvroRecordsToFile(filename=filename,
                               writer_schema=writer_schema)
    writer.write_records(records)

    return [filename]

  def _assert_same_tensors(self, expected_tensors, actual_tensors):

    logging.info("Expected tensors: {}".format(expected_tensors))
    logging.info("Actual tensors: {}".format(actual_tensors))

    assert len(expected_tensors) == len(actual_tensors), \
      "Expected {} pairs but " "got {} pairs".format(
          len(expected_tensors), len(actual_tensors))

    for name, actual_tensor in six.iteritems(actual_tensors):
      assert name in expected_tensors, "Expected key {} be present in {}"\
        .format(name, actual_tensors.keys())
      expected_tensor = expected_tensors[name]

      def assert_same_array(expected_array, actual_array):
        if np.issubdtype(actual_array.dtype, np.number):
          self.assertAllClose(expected_array, actual_array)
        else:
          self.assertAllEqual(expected_array, actual_array)

      if isinstance(actual_tensor, sparse_tensor.SparseTensorValue):
        self.assertAllEqual(expected_tensor.indices, actual_tensor.indices)
        assert_same_array(expected_tensor.values, actual_tensor.values)
      else:
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

  def _test_pass_dataset(self, writer_schema, record_data, expected_tensors,
                         features, batch_size, **kwargs):
    filenames = AvroDatasetTestBase._setup_files(writer_schema=writer_schema,
                                                 records=record_data)

    actual_dataset = avro_dataset.make_avro_dataset_v1(
        file_pattern=filenames, features=features, batch_size=batch_size,
        reader_schema=kwargs.get("reader_schema", ""),
        shuffle=kwargs.get("shuffle", None),
        num_epochs=kwargs.get("num_epochs", None))

    self._verify_output(expected_tensors=expected_tensors,
                        actual_dataset=actual_dataset)

  def _test_fail_dataset(self, writer_schema, record_data, features,
      batch_size, **kwargs):
    filenames = AvroDatasetTestBase._setup_files(writer_schema=writer_schema,
                                                 records=record_data)

    actual_dataset = avro_dataset.make_avro_dataset_v1(
        file_pattern=filenames, features=features,
        batch_size=batch_size,
        reader_schema=kwargs.get("reader_schema", ""),
        shuffle=kwargs.get("shuffle", None),
        num_epochs=kwargs.get("num_epochs", None))

    # Turn off any parallelism and random for testing
    config = config_pb2.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    with self.test_session(config=config) as sess:

      iterator = actual_dataset.make_initializable_iterator()
      next_element = iterator.get_next()
      sess.run(iterator.initializer)

      with self.assertRaises(OpError):
        sess.run(next_element)

