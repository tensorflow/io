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
"""FileDataset"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


@tf.function
def to_file(dataset, filename):
    """to_file"""
    resource = core_ops.io_file_init(filename)

    dataset = dataset.map(lambda e: (e, tf.constant(False)))
    dataset = dataset.concatenate(
        tf.data.Dataset.from_tensor_slices([tf.constant([], tf.string)]).map(
            lambda e: (e, tf.constant(True))
        )
    )
    dataset = dataset.map(
        lambda entry, final: core_ops.io_file_call(entry, final, resource)
    )
    dataset = dataset.map(tf.shape)

    return dataset.reduce(0, lambda x, y: x + y)
