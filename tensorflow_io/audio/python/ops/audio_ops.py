# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Audio Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
audio_ops = _load_library('_audio_ops.so')

class WAVDataset(data.Dataset):
  """A WAV Dataset
  """

  def __init__(self, filenames, batch=None):
    """Create a WAVDataset.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._data_input = audio_ops.wav_input(filenames)
    self._batch = 0 if batch is None else batch
    super(WAVDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return audio_ops.wav_dataset(
        self._data_input,
        self._batch,
        output_types=self.output_types,
        output_shapes=self.output_shapes)

  @property
  def output_shapes(self):
    return tuple([
        tensorflow.TensorShape([])]) if self._batch == 0 else tuple([
            tensorflow.TensorShape([None])])

  @property
  def output_classes(self):
    return tensorflow.Tensor


  @property
  def output_types(self):
    return tuple([dtypes.int16])
