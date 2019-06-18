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
"""PcapDataset"""
import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops as data_ops
from tensorflow_io import _load_library
pcap_ops = _load_library('_pcap_ops.so')


class PcapDataset(data_ops.Dataset):
  """A pcap Dataset. Pcap is a popular file format for capturing network packets.
  """

  def __init__(self, filenames, batch=None):
    """Create a pcap Reader.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.float64, tf.string]
    shapes = [
        tf.TensorShape([]), tf.TensorShape([])] if batch == 0 else [
            tf.TensorShape([None]), tf.TensorShape([None])]
    super(PcapDataset, self).__init__(
        pcap_ops.pcap_dataset,
        pcap_ops.pcap_input(filenames),
        batch, dtypes, shapes)
