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
"""Tests for TextDataset with stdin."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.text as text_io # pylint: disable=wrong-import-position

# Note: run the following:
#  tshark -T fields -e frame.number -e ip.dst -e ip.proto -r attack-trace.pcap | python stdin_test.py

def f(v):
  frame_number, ip_dst, ip_proto = tf.decode_csv(
      v, [[0], [''], [0]], field_delim='\t')
  return frame_number, ip_dst, ip_proto

text_dataset = text_io.TextDataset("file://-").map(f)

for (frame_number_value, ip_dst_value, ip_proto_value) in text_dataset:
  print(ip_dst_value.numpy())
