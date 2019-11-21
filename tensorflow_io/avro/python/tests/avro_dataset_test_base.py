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
import re
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

    def _assert_same_ranks(self, expected_data, actual_data_iterator):
        def __assert_same_ranks(expected_tensors, actual_tensors_iterator):
            assert len(expected_tensors) == len(actual_tensors_iterator), \
                "Expected {} pairs but " "got {} pairs".format(
                    len(expected_tensors), len(actual_tensors_iterator))

            for name, actual_tensor_iterator in actual_tensors_iterator.items():
                assert name in expected_tensors, "Expected key {} be present in {}" \
                    .format(name, actual_tensors_iterator.keys())
                expected_tensor = expected_tensors[name]
                # the shape is unknown for for parser key [%s]
                if re.search(r"\[(.*?)\]", name):
                    return
                if isinstance(actual_tensor_iterator, sparse_tensor.SparseTensor):
                    assert actual_tensor_iterator.shape.rank == expected_tensor.dense_shape.size
                else:
                    assert actual_tensor_iterator.shape.rank == len(expected_tensor.shape)

        if isinstance(expected_data, tuple):
            assert isinstance(actual_data_iterator, tuple), \
                "Found type {} but expected type {}".format(type(actual_data_iterator),
                                                            tuple)
            assert len(expected_data) == 2, \
                "Found {} components in expected dataset but must have {}" \
                    .format(len(expected_data), 2)

            assert len(actual_data_iterator) == 2, \
                "Found {} components in actual dataset but expected {}" \
                    .format(len(actual_data_iterator), 2)

            expected_features, expected_labels = expected_data
            actual_features_iter, actual_labels_iter = actual_data_iterator

            __assert_same_ranks(expected_features, actual_features_iter)
            __assert_same_ranks(expected_labels, actual_labels_iter)

        else:
            __assert_same_ranks(expected_data, actual_data_iterator)

    def _assert_same_data(self, expected_data, actual_data):
        if isinstance(expected_data, tuple):
            assert isinstance(actual_data, tuple), \
                "Found type {} but expected type {}".format(type(actual_data),
                                                            tuple)
            assert len(expected_data) == 2, \
                "Found {} components in expected dataset but must have {}" \
                    .format(len(expected_data), 2)

            assert len(actual_data) == 2, \
                "Found {} components in actual dataset but expected {}" \
                    .format(len(actual_data), 2)

            expected_features, expected_labels = expected_data
            actual_features, actual_labels = actual_data

            self._assert_same_tensors(expected_features, actual_features)
            self._assert_same_tensors(expected_labels, actual_labels)

        else:
            self._assert_same_tensors(expected_data, actual_data)

    def _assert_same_tensor(self, expected_tensor, actual_tensor):

        def assert_same_array(expected_array, actual_array):
            if np.issubdtype(actual_array.dtype, np.number):
                self.assertAllClose(expected_array, actual_array)
            else:
                self.assertAllEqual(expected_array, actual_array)

        if isinstance(actual_tensor, sparse_tensor.SparseTensorValue):
            self.assertAllEqual(expected_tensor.indices, actual_tensor.indices)
            self.assertAllEqual(expected_tensor.dense_shape, actual_tensor.dense_shape)
            assert_same_array(expected_tensor.values, actual_tensor.values)
        else:
            assert_same_array(expected_tensor, actual_tensor)

    def _assert_same_tensors(self, expected_tensors, actual_tensors):

        logging.info("Expected tensors: {}".format(expected_tensors))
        logging.info("Actual tensors: {}".format(actual_tensors))

        assert len(expected_tensors) == len(actual_tensors), \
            "Expected {} pairs but " "got {} pairs".format(
                len(expected_tensors), len(actual_tensors))

        for name, actual_tensor in actual_tensors.items():
            assert name in expected_tensors, "Expected key {} be present in {}" \
                .format(name, actual_tensors.keys())
            expected_tensor = expected_tensors[name]
            self._assert_same_tensor(actual_tensor=actual_tensor,
                                     expected_tensor=expected_tensor)

    def _verify_output(self, expected_data, actual_dataset, **kwargs):

        # Turn off any parallelism and random for testing
        config = config_pb2.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:

            iterator = actual_dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(iterator.initializer)

            for expected_datum in expected_data:
                self._assert_same_ranks(expected_data=expected_datum,
                                        actual_data_iterator=next_element)
                # Tried to leverage assertAllSame or assertAllEqual but that does
                # not handle string values properly
                self._assert_same_data(expected_data=expected_datum,
                                       actual_data=sess.run(next_element))

            if not kwargs.get("skip_out_of_range_error", False):
                with self.assertRaises(OutOfRangeError):
                    sess.run(next_element)

    def _test_pass_dataset(self, writer_schema, record_data, expected_data,
                           features, reader_schema, batch_size, **kwargs):
        filenames = AvroDatasetTestBase._setup_files(writer_schema=writer_schema,
                                                     records=record_data)

        actual_dataset = avro_dataset.make_avro_dataset_v1(
            filenames=filenames, reader_schema=reader_schema,
            features=features, batch_size=batch_size,
            shuffle=kwargs.get("shuffle", None),
            num_epochs=kwargs.get("num_epochs", None),
            label_keys=kwargs.get("label_keys", []))

        self._verify_output(expected_data=expected_data,
                            actual_dataset=actual_dataset,
                            **kwargs)

    def _test_fail_dataset(self, writer_schema, record_data, features,
                           reader_schema, batch_size, **kwargs):
        filenames = AvroDatasetTestBase._setup_files(writer_schema=writer_schema,
                                                     records=record_data)

        actual_dataset = avro_dataset.make_avro_dataset_v1(
            filenames=filenames, reader_schema=reader_schema,
            features=features, batch_size=batch_size,
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
