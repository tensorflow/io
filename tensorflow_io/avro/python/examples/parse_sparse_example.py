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
"""Example of parsing Avro data with sparse data."""

import tensorflow as tf

from tensorflow_io.avro.python.ops.parse_avro_record import parse_avro_record
from tensorflow_io.avro.python.utils.avro_serialization import AvroSerializer

schema = '''{"doc": "Sparse features.",
             "namespace": "com.linkedin.test.sparse",
             "type": "record",
             "name": "sparse_feature",
             "fields": [
                 {"name": "index", "type": "int"},
                 {"name": "sparse_type", "type": {
                     "type": "array",
                     "items": {
                         "type": "record",
                         "name": "sparse_triplet",
                         "fields": [
                             {"name": "index", "type": "long"},
                             {"name": "max_index", "type": "int"},
                             {"name": "value", "type": "float"}
                         ]
                     }
                 }}
             ]}'''

data = [{
    'index':
    0,
    'sparse_type': [{
        'index': 0,
        'max_index': 10,
        'value': 5.0
    }, {
        'index': 0,
        'max_index': 10,
        'value': 3.0
    }, {
        'index': 5,
        'max_index': 10,
        'value': 7.0
    }, {
        'index': 3,
        'max_index': 10,
        'value': 1.0
    }]
}, {
    'index':
    1,
    'sparse_type': [{
        'index': 0,
        'max_index': 10,
        'value': 2.0
    }, {
        'index': 9,
        'max_index': 10,
        'value': 1.0
    }]
}]

features = {
    'index':
    tf.FixedLenFeature([], tf.int32),
    'sparse_type':
    tf.SparseFeature(
        index_key='index',
        value_key='value',
        dtype=tf.float32,
        size=10,
        already_sorted=True)
}

if __name__ == '__main__':
    # Create a serializer
    serializer = AvroSerializer(schema)

    # Serialize data into a batch
    serialized = [serializer.serialize(datum) for datum in data]

    with tf.Session() as sess:
        # Variable to feed the serialized string
        input_str = tf.placeholder(tf.string)
        # Use the parse function
        tensors = parse_avro_record(input_str, schema, features)
        print(sess.run(tensors, feed_dict={input_str: serialized}))
