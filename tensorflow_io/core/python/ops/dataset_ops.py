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
"""Dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.audio.python.ops import audio_ops

def create_dataset_class():
  """create_dataset_class"""
  dataset_class = tf.compat.v2.data.Dataset

  def from_audio(_, filename, **kwargs):
    return audio_ops.WAVDataset(filename, **kwargs)

  setattr(dataset_class, "from_audio", classmethod(from_audio))

  return dataset_class

Dataset = create_dataset_class() # pylint: disable=invalid-name
