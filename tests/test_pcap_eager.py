# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""
Test PcapDataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_io.pcap as pcap_io # pylint: disable=wrong-import-position

if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()


def test_pcap_input():
  """test_pcap_input
  """
  print("Testing PcapDataset")
  pcap_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_pcap", "http.pcap")
  file_url = "file://" + pcap_filename
  url_filenames = [file_url]
  dataset = pcap_io.PcapDataset(url_filenames, batch=1)

  packets_total = 0
  for v in dataset:
    (packet_timestamp, packet_data) = v
    if packets_total == 0:
      assert packet_timestamp.numpy()[0] == 1084443427.311224 # we know this is the correct value in the test pcap file
      assert len(packet_data.numpy()[0]) == 62 # we know this is the correct packet data buffer length in the test pcap file
    packets_total += 1
  assert packets_total == 43 # we know this is the correct number of packets in the test pcap file

if __name__ == "__main__":
  test.main()
