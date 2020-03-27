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
"""LibSVMDataset"""

import tensorflow as tf
from tensorflow_io.core.python.experimental.text_ops import decode_libsvm


class LibSVMIODataset(tf.data.Dataset):
    """LibSVMIODataset"""

    def __init__(
        self,
        filename,
        num_features,
        dtype=None,
        label_dtype=None,
        compression_type="",
        internal=True,
    ):
        if not internal:
            raise ValueError(
                "LibSVMIODataset constructor is private; please use one "
                "of the factory methods instead (e.g., "
                "IODataset.from_pcap())"
            )
        with tf.name_scope("LibSVMIODataset"):
            dataset = tf.data.TextLineDataset(
                filename, compression_type=compression_type
            )
            dataset = dataset.batch(1).map(
                lambda e: decode_libsvm(e, num_features, dtype, label_dtype)
            )

            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
