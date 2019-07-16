# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Text Input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
from sklearn.model_selection import train_test_split

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.frame as frame_io # pylint: disable=wrong-import-position

def test_from_csv():
  """test from https://www.tensorflow.org/beta/tutorials/keras/feature_columns
  """
  sample = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_text", "heart.csv")
  with open(sample, "r") as f:
    entries = [line for line in csv.reader(f)]
  sample = "file://" + sample

  df = frame_io.DataFrame.from_csv(sample)

  train, test = df.split(lambda x: train_test_split(x, test_size=0.2))
  train, val = train.split(lambda x: train_test_split(x, test_size=0.2))

  print(len(train), 'train examples')
  print(len(val), 'validation examples')
  print(len(test), 'test examples')
  assert len(train) == 193
  assert len(val) == 49
  assert len(test) == 61

  def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    # dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
      ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

  batch_size = 5 # A small batch sized is used for demonstration purposes
  train_ds = df_to_dataset(train, batch_size=batch_size)
  val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
  test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

  train_ds_v = [v for v in train_ds]
  val_ds_v = [v for v in val_ds]
  test_ds_v = [v for v in test_ds]
  assert len(train_ds_v) == 39
  assert len(val_ds_v) == 10
  assert len(test_ds_v) == 13

  # Validate data
  assert ",".join(entries[0]) == ",".join(df.columns)
  i = 1
  for entry in tf.data.Dataset.from_tensor_slices((dict(df))):
    assert entries[i][0] == str(entry['age'].numpy())
    assert entries[i][1] == str(entry['sex'].numpy())
    assert entries[i][12] == str(entry['thal'].numpy())
    assert entries[i][13] == str(entry['target'].numpy())
    i += 1
  assert i == len(entries)

if __name__ == "__main__":
  test.main()
