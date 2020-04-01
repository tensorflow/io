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
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.framework import ops
from tensorflow.python.ops import parsing_ops
from tensorflow_io.core.python.experimental.make_avro_record_dataset import make_avro_record_dataset

from tensorflow_io.avro.python.tests.avro_dataset_test_base import AvroDatasetTestBase


class MakeAvroRecordDatasetTest(AvroDatasetTestBase):

    def _test_pass_dataset(self, writer_schema, record_data, expected_data,
                           features, reader_schema, batch_size, **kwargs):
        filenames = AvroDatasetTestBase._setup_files(writer_schema=writer_schema,
                                                     records=record_data)

        actual_dataset = make_avro_record_dataset(
            file_pattern=filenames,
            features=features,
            batch_size=batch_size,
            reader_schema=reader_schema,
            shuffle=kwargs.get("shuffle", None),
            num_epochs=kwargs.get("num_epochs", None))

        self._verify_output(expected_data=expected_data,
                            actual_dataset=actual_dataset)

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
        expected_data = [
            {"int_value": ops.convert_to_tensor([0, 1])},
            {"int_value": ops.convert_to_tensor([2])}
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_fixed_length_list(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  }
              ]}"""
        record_data = [
            {"int_list": [0, 1, 2]},
            {"int_list": [3, 4, 5]},
            {"int_list": [6, 7, 8]}
        ]
        features = {
            "int_list[*]": parsing_ops.FixedLenFeature([3], tf_types.int32)
        }
        expected_data = [
            {"int_list[*]": ops.convert_to_tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])}
        ]

        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)
