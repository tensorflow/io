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

import os
import tempfile

from tensorflow.python.framework import test_util
from tensorflow.python.data.kernel_tests import test_base
from tensorflow_io.avro.python.ops.avro_record_dataset import AvroRecordDatasetV2
from tensorflow_io.avro.python.utils.avro_serialization import \
    AvroRecordsToFile


@test_util.run_all_in_graph_and_eager_modes
class AvroRecordDatasetTest(test_base.DatasetTestBase):

    @staticmethod
    def _setup_files(writer_schema, records):
        # Write test records into temporary output directory
        filename = os.path.join(tempfile.mkdtemp(), "test.avro")
        writer = AvroRecordsToFile(filename=filename,
                                   writer_schema=writer_schema)
        writer.write_records(records)

        return [filename]

    def test_wout_reader_schema(self):
        writer_schema = """{
              "type": "record",
              "name": "dataTypes",
              "fields": [
                  {
                     "name":"string_value",
                     "type":"string"
                  }
              ]}"""
        record_data = [
            {
                "string_value": ""
            },
            {
                "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"
            },
            {
                "string_value": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789"
            }
        ]
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '50'

        filenames = AvroRecordDatasetTest._setup_files(writer_schema=writer_schema, records=record_data)
        dataset = AvroRecordDatasetV2(filenames)
        print(dataset.take(10))
