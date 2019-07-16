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
"""TextInput/TextOutput."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import csv
import numpy as np

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops as data_ops
from tensorflow_io.core.python.ops import core_ops as text_ops

def save_text(dataset, filename):
  """Save Dataset to disk.

  Args:
    dataset: A TextDataset to be saved.
    filename: A `tf.string` tensor containing filename.
  """
  return text_ops.text_dataset_output(dataset._variant_tensor, filename) # pylint: disable=protected-access


def save_csv(dataset, filename):
  """Save Dataset to disk.

  Args:
    dataset: A Dataset to be saved.
    filename: A `tf.string` tensor containing filename.
  """
  return text_ops.csv_dataset_output(dataset._variant_tensor, filename) # pylint: disable=protected-access


class TextDataset(data_ops.Dataset):
  """A Text Dataset"""

  def __init__(self, filename, batch=None):
    """Create a Text Reader.

    Args:
      filename: A `tf.string` tensor containing one or more filenames.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.string]
    shapes = [
        tf.TensorShape([])] if batch == 0 else [
            tf.TensorShape([None])]
    fn = text_ops.text_stream_dataset if (
        filename == 'file://-') else text_ops.text_dataset
    data_input = text_ops.text_stream_input(filename) if (
        filename == 'file://-') else text_ops.text_input(
            filename, ["none", "gz"])
    super(TextDataset, self).__init__(
        fn,
        data_input,
        batch, dtypes, shapes)


class TextOutputSequence(object):
  """TextOutputSequence"""

  def __init__(self, filenames):
    """Create a `TextOutputSequence`.
    """
    self._filenames = filenames
    self._resource = text_ops.text_output_sequence(destination=filenames)

  def setitem(self, index, item):
    text_ops.text_output_sequence_set_item(self._resource, index, item)


def _infer_dtype(val):
  """_infer_dtype"""
  try:
    val = ast.literal_eval(val)
  except (SyntaxError, ValueError):
    return tf.string
  if isinstance(val, int):
    if np.int32(val) == val:
      return tf.int32
    elif np.int64(val) == val:
      return tf.int64
  elif isinstance(val, float):
    if np.float32(val) == val:
      return tf.float32
    elif np.float64(val) == val:
      return tf.float64
  return tf.string


def from_csv(filename, header=0):
  """Read csv to Dataset

  NOTE: Experimental and eager only!

  Args:
    filename: A `tf.string` tensor containing filename.
  """
  if not tf.executing_eagerly():
    raise NotImplementedError("from_csv only supports eager mode")
  dataset = TextDataset(filename)
  columns = None
  if header is not None:
    if header != 0:
      raise NotImplementedError(
          "from_csv only supports header=0 or header=None for now")
    # Read first linea as name
    columns = list(
        csv.reader([line.numpy().decode() for line in dataset.take(1)]))[0]
    dataset = dataset.skip(1)
  entries = list(
      csv.reader([line.numpy().decode() for line in dataset.take(1)]))[0]
  if columns is None:
    columns = [i for (i, _) in enumerate(entries)]
  dtypes = [_infer_dtype(column) for column in entries]
  specs = [
      tf.TensorSpec(tf.TensorShape([]), dtype, column) for (
          column, dtype) in zip(columns, dtypes)]

  record_defaults = [tf.zeros(spec.shape, spec.dtype) for spec in specs]
  return tf.data.experimental.CsvDataset(
      filename, record_defaults, header=(header is not None)), specs
