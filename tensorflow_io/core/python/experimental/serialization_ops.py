# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Serialization Ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

# _NamedTensorSpec allows adding a `named` key while traversing,
# so that it is possible to build up the `/R/Foo` JSON Pointers.
class _NamedTensorSpec(tf.TensorSpec):
  """_NamedTensorSpec"""
  def named(self, named=None):
    if named is not None:
      self._named = named
    return self._named

# named_spec updates named field for JSON Pointers while traversing.
def named_spec(specs, name=''):
  """named_spec"""
  if isinstance(specs, _NamedTensorSpec):
    specs.named(name)
    return

  if isinstance(specs, dict):
    for k in specs.keys():
      named_spec(specs[k], "{}/{}".format(name, k))
    return

  for k, _ in enumerate(specs):
    named_spec(specs[k], "{}/{}".format(name, k))
  return


def decode_json(json, specs, name=None):
  """
  Decode JSON string into Tensors.

  TODO: support batch (1-D) input

  Args:
    json: A String Tensor. The JSON strings to decode.
    specs: A structured TensorSpecs describing the signature
      of the JSON elements.
    name: A name for the operation (optional).

  Returns:
    A structured Tensors.
  """
  # Make a copy of specs to keep the original specs
  named = tf.nest.map_structure(lambda e: _NamedTensorSpec(e.shape, e.dtype), specs)
  named_spec(named)
  named = tf.nest.flatten(named)
  names = [e.named() for e in named]
  shapes = [e.shape for e in named]
  dtypes = [e.dtype for e in named]

  values = core_ops.io_decode_json(json, names, shapes, dtypes, name=name)
  return tf.nest.pack_sequence_as(specs, values)
