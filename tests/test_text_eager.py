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
import tempfile
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.text as text_io # pylint: disable=wrong-import-position

def test_text_input():
  """test_text_input
  """
  text_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_text", "lorem.txt")
  with open(text_filename, 'rb') as f:
    lines = [line.strip() for line in f]
  text_filename = "file://" + text_filename

  gzip_text_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_text", "lorem.txt.gz")
  gzip_text_filename = "file://" + gzip_text_filename

  lines = lines * 3
  filenames = [text_filename, gzip_text_filename, text_filename]
  text_dataset = text_io.TextDataset(filenames, batch=2)
  i = 0
  for v in text_dataset:
    assert lines[i] == v.numpy()[0]
    i += 1
    if i < len(lines):
      assert lines[i] == v.numpy()[1]
      i += 1
  assert i == len(lines)


def test_text_output_sequence():
  """Test case based on fashion mnist tutorial"""
  fashion_mnist = tf.keras.datasets.fashion_mnist
  ((train_images, train_labels),
   (test_images, _)) = fashion_mnist.load_data()

  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  train_images = train_images / 255.0
  test_images = test_images / 255.0

  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(train_images, train_labels, epochs=5)

  class OutputCallback(tf.keras.callbacks.Callback):
    """OutputCallback"""
    def __init__(self, filename, batch_size):
      self._sequence = text_io.TextOutputSequence(filename)
      self._batch_size = batch_size

    def on_predict_batch_end(self, batch, logs=None):
      index = batch * self._batch_size
      for outputs in logs['outputs']:
        for output in outputs:
          self._sequence.setitem(index, class_names[np.argmax(output)])
          index += 1

  f, filename = tempfile.mkstemp()
  os.close(f)
  # By default batch size is 32
  output = OutputCallback(filename, 32)
  predictions = model.predict(test_images, callbacks=[output])
  with open(filename) as f:
    lines = [line.strip() for line in f]
  predictions = [class_names[v] for v in np.argmax(predictions, axis=1)]
  assert len(lines) == len(predictions)
  for line, prediction in zip(lines, predictions):
    assert line == prediction

def test_text_output():
  """test_text_output"""
  text_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_text", "lorem.txt")
  with open(text_filename, 'rb') as f:
    lines = [line.strip() for line in f]
  text_filename = "file://" + text_filename

  f, filename = tempfile.mkstemp()
  os.close(f)

  df = text_io.TextDataset(text_filename)
  df = df.take(5)
  text_io.save_text(df, filename)

  with open(filename, 'rb') as f:
    saved_lines = [line.strip() for line in f]
  i = 0
  for line in saved_lines:
    assert lines[i] == line
    i += 1
  assert i == 5

def test_csv_output():
  """test_csv_output"""
  csv_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_text", "sample.csv")
  with open(csv_filename, 'rb') as f:
    lines = [line.strip() for line in f]
  csv_filename = "file://" + csv_filename

  f, filename = tempfile.mkstemp()
  os.close(f)

  df = tf.data.experimental.CsvDataset(csv_filename, [0, 0, 0])
  df = df.take(5)
  text_io.save_csv(df, filename)

  with open(filename, 'rb') as f:
    saved_lines = [line.strip() for line in f]
  i = 0
  for line in saved_lines:
    assert lines[i] == line
    i += 1
  assert i == 5

def test_from_csv():
  """test from_csv with sample csv file.

  sample1.csv
  "col1",col2,"col 3","col 4",col5
  15,10000000000,3.0,4.0e30,"col 5 string 1"
  30,20000000000,6.0,8.0e30,"col 5 string 2"

  sample2.csv
  15,10000000000,3.0,4.0e30,"col 5 string 1"
  30,20000000000,6.0,8.0e30,"col 5 string 2"

  header=None will use range(width) as column names.
  """
  expected = [
      [np.int32(15),
       np.int64(10000000000),
       np.float32(3.0),
       np.float64(4.0e30),
       b"col 5 string 1"],
      [np.int32(30),
       np.int64(20000000000),
       np.float32(6.0),
       np.float64(8.0e30),
       b"col 5 string 2"]]

  sample1 = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_text", "sample1.csv")
  sample1 = "file://" + sample1

  df1, _ = text_io.from_csv(sample1)

  sample2 = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_text", "sample2.csv")
  sample2 = "file://" + sample2

  df2, _ = text_io.from_csv(sample2, header=None)

  for df in [df1, df2]:
    line = 0
    for records in df:
      assert len(records) == len(expected[line])
      for i, record in enumerate(records):
        if isinstance(expected[line][i], bytes):
          assert record.dtype == tf.string
          assert record.numpy() == expected[line][i]
        else:
          assert record.dtype == expected[line][i].dtype
          assert record.numpy() == expected[line][i]
      line += 1

if __name__ == "__main__":
  test.main()
