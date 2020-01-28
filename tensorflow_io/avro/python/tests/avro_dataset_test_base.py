# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import os
import tempfile

from tensorflow.python.framework import test_util, sparse_tensor
from tensorflow.python.framework.errors import OpError, OutOfRangeError
from tensorflow.python.data.kernel_tests import test_base
from tensorflow_io.core.python.experimental.avro_dataset_ops import make_avro_dataset
from tensorflow_io.avro.python.utils.avro_serialization import AvroRecordsToFile


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

    def assertDataEqual(self, expected, actual):

        def _assertEqual(_expected, _actual):
            for name, datum in _expected.items():
                self.assertValuesEqual(expected=datum, actual=_actual[name])

        if isinstance(expected, tuple):
            assert isinstance(expected, tuple), \
                "Found type {} but expected type {}".format(type(actual), tuple)
            assert len(expected) == 2, \
                "Found {} components in expected dataset but must have {}" \
                    .format(len(expected), 2)

            assert len(actual) == 2, \
                "Found {} components in actual dataset but expected {}" \
                    .format(len(actual), 2)

            expected_features, expected_labels = expected
            actual_features, actual_labels = actual

            _assertEqual(expected_features, actual_features)
            _assertEqual(expected_labels, actual_labels)

        else:
            _assertEqual(expected, actual)


    def _verify_output(self, expected_data, actual_dataset):

        next_data = iter(actual_dataset)

        for expected in expected_data:
            self.assertDataEqual(expected=expected, actual=next(next_data))

    def _test_pass_dataset(self, writer_schema, record_data, expected_data,
                           features, reader_schema, batch_size, **kwargs):
        filenames = AvroDatasetTestBase._setup_files(writer_schema=writer_schema,
                                                     records=record_data)

        actual_dataset = make_avro_dataset(
            filenames=filenames, reader_schema=reader_schema,
            features=features, batch_size=batch_size,
            shuffle=kwargs.get("shuffle", None),
            num_epochs=kwargs.get("num_epochs", None),
            label_keys=kwargs.get("label_keys", []))

        self._verify_output(expected_data=expected_data,
                            actual_dataset=actual_dataset)

    def _test_fail_dataset(self, writer_schema, record_data, features,
                           reader_schema, batch_size, **kwargs):
        filenames = AvroDatasetTestBase._setup_files(writer_schema=writer_schema,
                                                     records=record_data)

        actual_dataset = make_avro_dataset(
            filenames=filenames, reader_schema=reader_schema,
            features=features, batch_size=batch_size,
            shuffle=kwargs.get("shuffle", None),
            num_epochs=kwargs.get("num_epochs", None))

        next_data = iter(actual_dataset)

        with self.assertRaises(OpError):
            next(next_data)
