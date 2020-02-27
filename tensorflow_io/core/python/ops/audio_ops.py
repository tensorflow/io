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
"""Audio Ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_io.core.python.ops import core_ops

def decode_mp3(contents, desired_channels=-1, desired_samples=-1, name=None):
  """
  Decode MP3 data to an int16 tensor.

  Args:
    contents: A `Tensor` of type `string`. 0-D. The encoded MP3.
    desired_channels: An integer determining how many channels are in the output.
        If this number is greater than the number of channels available in the source,
        they are duplicated. If it is smaller, only the first is used.
    desired_samples: An integer determining how many samples are in the ouput (per channel).
        If this number is greater than the number of samples available in the source,
        the output is 0-padded on the right. If it is smaller, the output is truncated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int16` and shape of `[channels, samples]` and a scalar `Tensor`
    of type `int32` containing the sample rate of the MP3.
  """
  # TODO do I really just have to pass in the name here?
  return core_ops.io_decode_mp3(contents, desired_channels, desired_samples, name=name)
