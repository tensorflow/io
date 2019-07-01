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
"""JSONDataset"""
import json
import tensorflow as tf
# from tensorflow_io.core.python.ops import data_ops as data_ops
# from tensorflow_io.core.python.ops import core_ops as json_ops
# from tensorflow_io import _load_library
# json_ops = _load_library('_json_ops.so')


# class JSONDataset(data_ops.Dataset):
#   """A JSONLabelDataset. JSON (JavaScript Object Notation) is a lightweight data-interchange format.
#   """

#   def __init__(self, filenames, batch=None):
#     """Create a JSONLabelDataset.

#     Args:
#       filenames: A `tf.string` tensor containing one or more filenames.
#     """
    # batch = 0 if batch is None else batch
    # dtypes = [tf.float64, tf.string]
    # shapes = [
    # tf.TensorShape([]), tf.TensorShape([])] if batch == 0 else [
    # tf.TensorShape([None]), tf.TensorShape([None])]
    # super(JSONDataset, self).__init__(
    # json_ops.json_dataset,
    # json_ops.json_input(filenames),
    # batch, dtypes, shapes)


def JSONDataset(filenames, columns=None):
  """Start with JSON parser in python to add tests."""
  jsondataset = []
  jsondataset = JSONParser(filenames, columns)
  return tf.data.Dataset.from_tensor_slices(jsondataset)


def JSONParser(filenames, columns):
  """JSON parser in Python for testing."""
  with open(filenames) as json_file:
    data = json.load(json_file)
  dataset = []
  for sample in data:
    sampledata = []
    if columns is None:
      for key in sorted(sample):
        sampledata.append(sample[key])
    else:
      for key in columns:
        if  key in sample:
          sampledata.append(sample[key])
    dataset.append(sampledata)
  return dataset
