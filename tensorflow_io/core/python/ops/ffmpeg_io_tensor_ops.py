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
"""FFmpegIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import uuid
import warnings

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops

class _IOTensorIterableNextFunction(object):
  def __init__(self, resource, component, function, shape, dtype, capacity):
    self._resource = resource
    self._component = component
    self._function = function
    self._shape = shape
    self._dtype = dtype
    self._capacity = capacity
  def __call__(self):
    return self._function(
        self._resource, component=self._component,
        capacity=self._capacity, shape=self._shape, dtype=self._dtype)

class _IOTensorIterablePartitionedFunction(object):
  """PartitionedFunction will translate call to cached Function call"""
  # function: next call of the iterable
  def __init__(self, function, shape):
    self._function = function
    self._partitions = []
    self._length = None
    self._slice_suffix_start = [0 for _ in shape[1:]]
    self._slice_suffix_size = [e for e in shape[1:]]

  def __call__(self, start, stop):
    while self._length is None:
      # if stop is not None then resolved partitions have to cover stop
      # if stop is None then all partitions has to be resolved
      if stop is not None:
        if stop <= sum([e.shape[0] for e in self._partitions]):
          break
      # resolve one step
      partition = self._function()
      if partition.shape[0] == 0:
        self._length = sum([e.shape[0] for e in self._partitions])
      else:
        self._partitions.append(partition)
      partitions_indices = tf.cumsum(
          [e.shape[0] for e in self._partitions]).numpy().tolist()
      self._partitions_start = list([0] + partitions_indices[:-1])
      self._partitions_stop = partitions_indices
    length = self._partitions_stop[-1]
    index = slice(start, stop)
    start, stop, _ = index.indices(length)
    if start >= length:
      raise IndexError("index %s is out of range" % index)
    indices_start = tf.math.maximum(self._partitions_start, start)
    indices_stop = tf.math.minimum(self._partitions_stop, stop)
    indices_hit = tf.math.less(indices_start, indices_stop)
    indices = tf.squeeze(tf.compat.v2.where(indices_hit), [1])
    return self._partitions_read(indices_start, indices_stop, indices)

  def _partitions_read(self, indices_start, indices_stop, indices):
    """_partitions_read"""
    items = []
    # TODO: change to tf.while_loop
    for index in indices:
      slice_start = indices_start[index] - self._partitions_start[index]
      slice_size = indices_stop[index] - indices_start[index]
      slice_start = [slice_start] + self._slice_suffix_start
      slice_size = [slice_size] + self._slice_suffix_size
      item = tf.slice(self._partitions[index], slice_start, slice_size)
      items.append(item)
    return tf.concat(items, axis=0)
  def _partitions_length(self):
    """_partitions_length"""
    while self._length is None:
      # resolve until length is available
      partition = self._function()
      if partition.shape[0] == 0:
        self._length = sum([e.shape[0] for e in self._partitions])
      else:
        self._partitions.append(partition)
      partitions_indices = tf.cumsum(
          [e.shape[0] for e in self._partitions]).numpy().tolist()
      self._partitions_start = list([0] + partitions_indices[:-1])
      self._partitions_stop = partitions_indices
    return self._length


class FFmpegBaseIOTensor(io_tensor_ops._IOTensor): # pylint: disable=protected-access
  """FFmpegAudioIOTensor"""
  def __init__(self,
               spec, function,
               internal=False):
    with tf.name_scope("FFmpegBaseIOTensor"):
      self._function = function
      super(FFmpegBaseIOTensor, self).__init__(
          spec, internal=internal)

  @property
  def shape(self):
    return self.spec.shape
  @property
  def dtype(self):
    return self.spec.dtype
  def to_tensor(self, **kwargs):
    with tf.name_scope(kwargs.get("name", "IOToTensor")):
      return self.__getitem__(slice(None, None))
  def __getitem__(self, key):
    """Returns the specified piece of this IOTensor."""
    # Find out the indices based on length and key,
    # based on python slice()'s indices method:
    index = key if isinstance(key, slice) else slice(key, key + 1)
    items = self._function(start=index.start, stop=index.stop)
    return tf.squeeze(items, axis=[0]) if items.shape[0] == 1 else items
  def __len__(self):
    """Returns the total number of items of this IOTensor."""
    return self._function._partitions_length() # pylint: disable=protected-access

class FFmpegVideoIOTensor(FFmpegBaseIOTensor): # pylint: disable=protected-access
  """FFmpegVideoIOTensor"""
  def __init__(self,
               spec, function,
               internal=False):
    with tf.name_scope("FFmpegVideoIOTensor"):
      super(FFmpegVideoIOTensor, self).__init__(
          spec, function, internal=internal)
  def __getitem__(self, key):
    warnings.warn(
        "Indexed access of video may consume much resource, "
        "please consider using VideoDataset instead",
        Warning if sys.version_info[0] < 3 else ResourceWarning)
    return super(FFmpegVideoIOTensor, self).__getitem__(key)
  def __len__(self):
    warnings.warn(
        "Indexed access of video may consume much resource, "
        "please consider using VideoDataset instead",
        Warning if sys.version_info[0] < 3 else ResourceWarning)
    return super(FFmpegVideoIOTensor, self).__len__()


class FFmpegAudioIOTensor(FFmpegBaseIOTensor): # pylint: disable=protected-access
  """FFmpegAudioIOTensor"""
  def __init__(self,
               spec, function, rate,
               internal=False):
    with tf.name_scope("FFmpegAudioIOTensor"):
      self._rate = rate
      super(FFmpegAudioIOTensor, self).__init__(
          spec, function, internal=internal)
  @io_tensor_ops._IOTensorMeta # pylint: disable=protected-access
  def rate(self):
    return self._rate

class FFmpegSubtitleIOTensor(FFmpegBaseIOTensor): # pylint: disable=protected-access
  """FFmpegSubtitleIOTensor"""
  def __init__(self,
               spec, function,
               internal=False):
    with tf.name_scope("FFmpegSubtitleIOTensor"):
      super(FFmpegSubtitleIOTensor, self).__init__(
          spec, function, internal=internal)

class FFmpegIOTensor(io_tensor_ops._CollectionIOTensor): # pylint: disable=protected-access
  """FFmpegIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("FFmpegIOTensor") as scope:
      from tensorflow_io.core.python.ops import ffmpeg_ops
      resource, columns = ffmpeg_ops.ffmpeg_iterable_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      columns = [column.decode() for column in columns.numpy().tolist()]
      elements = []
      for column in columns:
        shape, dtype, rate = ffmpeg_ops.ffmpeg_iterable_spec(resource, column)
        shape = tf.TensorShape([None if e < 0 else e for e in shape.numpy()])
        dtype = tf.as_dtype(dtype.numpy())
        spec = tf.TensorSpec(shape, dtype, column)
        if column.startswith("a:"):
          function = _IOTensorIterableNextFunction(
              resource, column,
              ffmpeg_ops.ffmpeg_iterable_next,
              shape, dtype, capacity=4096)
          rate = rate.numpy()
          elements.append(FFmpegAudioIOTensor(
              spec, _IOTensorIterablePartitionedFunction(function, shape),
              rate, internal=internal))
        elif column.startswith("s:"):
          function = _IOTensorIterableNextFunction(
              resource, column,
              ffmpeg_ops.ffmpeg_iterable_next,
              shape, dtype, capacity=4096)
          elements.append(FFmpegSubtitleIOTensor(
              spec, _IOTensorIterablePartitionedFunction(function, shape),
              internal=internal))
        else:
          function = _IOTensorIterableNextFunction(
              resource, column,
              ffmpeg_ops.ffmpeg_iterable_next,
              shape, dtype, capacity=1)
          elements.append(FFmpegVideoIOTensor(
              spec, _IOTensorIterablePartitionedFunction(function, shape),
              internal=internal))
      spec = tuple([e.spec for e in elements])
      super(FFmpegIOTensor, self).__init__(
          spec, columns, elements,
          internal=internal)
