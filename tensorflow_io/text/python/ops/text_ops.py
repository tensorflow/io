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
from tensorflow_io.core.python.ops import data_ops
from tensorflow_io.core.python.ops import core_ops

def read_text(filename, **kwargs):
  """read_text"""
  memory = kwargs.get("memory", "")
  offset = kwargs.get("offset", 0)
  length = kwargs.get("length", -1)
  return core_ops.read_text(
      filename, offset=offset, length=length, memory=memory)

def save_text(dataset, filename):
  """Save Dataset to disk.

  Args:
    dataset: A TextDataset to be saved.
    filename: A `tf.string` tensor containing filename.
  """
  return core_ops.text_dataset_output(dataset._variant_tensor, filename) # pylint: disable=protected-access


def save_csv(dataset, filename):
  """Save Dataset to disk.

  Args:
    dataset: A Dataset to be saved.
    filename: A `tf.string` tensor containing filename.
  """
  return core_ops.csv_dataset_output(dataset._variant_tensor, filename) # pylint: disable=protected-access


def re2_full_match(input, pattern): # pylint: disable=redefined-builtin
  """Extract regex groups

  Args:
    dataset: A `tf.string` tensor
    pattern: A pattern string.
  """
  return core_ops.re2_full_match(input, pattern)


class TextDataset(data_ops.BaseDataset):
  """A Text Dataset"""

  def __init__(self, filename, **kwargs):
    """Create a Text Reader.

    Args:
      filename: A string containing filename to read.
    """
    dtype = tf.string
    shape = tf.TensorShape([None])

    capacity = kwargs.get("capacity", 65536)

    if filename.startswith("file://-") or filename.startswith("file://0"):
      dataset = data_ops.BaseDataset.range(1).map(
          lambda length: core_ops.read_text(filename, memory="", offset=0, length=length)
      )
    else:
      filesize = tf.io.gfile.GFile(filename).size()
      # capacity is the rough length for each split
      entry_offset = list(range(0, filesize, capacity))
      entry_length = [
          min(capacity, filesize - offset) for offset in entry_offset]
      dataset = data_ops.BaseDataset.from_tensor_slices(
          (
              tf.constant(entry_offset, tf.int64),
              tf.constant(entry_length, tf.int64)
          )
      ).map(lambda offset, length: core_ops.read_text(
          filename, memory="",
          offset=offset, length=length))
    self._dataset = dataset

    super(TextDataset, self).__init__(
        self._dataset._variant_tensor, [dtype], [shape]) # pylint: disable=protected-access

class TextOutputSequence(object):
  """TextOutputSequence"""

  def __init__(self, filenames):
    """Create a `TextOutputSequence`.
    """
    self._filenames = filenames
    self._resource = core_ops.text_output_sequence(destination=filenames)

  def setitem(self, index, item):
    core_ops.text_output_sequence_set_item(self._resource, index, item)


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
  dataset = TextDataset(filename).apply(tf.data.experimental.unbatch())
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
