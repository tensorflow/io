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
from tensorflow_io.core.python.ops import pcap_dataset_ops

class PcapDataset(pcap_dataset_ops.PcapIODataset):
  """A pcap Dataset. Pcap is a popular file format for capturing network packets.
  """

  def __init__(self, filename):
    """Create a pcap Reader.

    Args:
      filename: A `tf.string` tensor containing filename.
    """
    super(PcapDataset, self).__init__(filename)
