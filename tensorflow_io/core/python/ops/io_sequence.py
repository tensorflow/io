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
"""IOSequence is a subclass of tf.keras.util.Sequence"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

class IOSequence(tf.keras.utils.Sequence):
  """IOSequence

  An `IOSequence` is a subclass of tf.keras.util.Sequence
  that could be used to pass collections of IOTensor to tf.keras
  for training purposes.

  Since IOTensor is always repeatable, multiple runs will not change
  the underlying data, it could avoid the cases where steaming dataset
  are not able to be used in tf.keras due to unreliable re-run.

  Example:

  ```python

  x = tfio.IOTensor.from_tensor(tf.reshape(
      tf.image.convert_image_dtype(x_train, tf.float32), [-1, 28, 28]))
  y = tfio.IOTensor.from_tensor(y_train)

  d_train = tfio.IOSequence(batch_size=1, x=x, y=y)

  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(d_train, epochs=5)
  ```
  """

  def __init__(self, batch_size, x, y):
    self.batch_size = batch_size
    self.x = x
    self.y = y

  def __len__(self):
    return int(np.ceil(len(self.x) / float(self.batch_size)))

  def __getitem__(self, idx):
    batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    if self.batch_size == 1:
      return tf.expand_dims(batch_x, 0), tf.expand_dims(batch_y, 0)
    return batch_x, batch_y
