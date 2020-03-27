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
"""TIFFIODataset"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class TIFFIODataset(tf.data.Dataset):
    """TIFFIODataset"""

    def __init__(self, filename, internal=True):
        if not internal:
            raise ValueError(
                "TIFFIODataset constructor is private; please use one "
                "of the factory methods instead (e.g., "
                "IODataset.from_pcap())"
            )
        with tf.name_scope("TIFFIODataset"):
            content = tf.io.read_file(filename)
            _, dtype = core_ops.io_decode_tiff_info(content)
            # use dtype's rank to find out the number of elements
            dataset = tf.data.Dataset.range(tf.cast(tf.shape(dtype)[0], tf.int64))
            dataset = dataset.map(lambda index: core_ops.io_decode_tiff(content, index))

            self._dataset = dataset
            self._content = content
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
