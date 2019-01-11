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
"""PubSub Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
pubsub_ops = _load_library('_pubsub_ops.so')

class PubSubDataset(data.Dataset):
  """A PubSub Dataset that consumes the message.
  """

  def __init__(self,
               subscriptions,
               server=None,
               eof=False,
               timeout=1000):
    """Create a PubSubDataset.

    Args:
      subscriptions: A `tf.string` tensor containing one or more subscriptions.
      server: The pubsub server.
      eof: If True, the pubsub reader will stop on EOF.
      timeout: The timeout value for the PubSub to wait
               (in millisecond).
    """
    self._subscriptions = tensorflow.convert_to_tensor(
        subscriptions, dtype=dtypes.string, name="subscriptions")
    self._server = tensorflow.convert_to_tensor(
        server, dtype=dtypes.string, name="server")
    self._eof = tensorflow.convert_to_tensor(eof, dtype=dtypes.bool, name="eof")
    self._timeout = tensorflow.convert_to_tensor(
        timeout, dtype=dtypes.int64, name="timeout")
    super(PubSubDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return pubsub_ops.pub_sub_dataset(self._subscriptions, self._server,
                                      self._eof, self._timeout)

  @property
  def output_classes(self):
    return tensorflow.Tensor

  @property
  def output_shapes(self):
    return tensorflow.TensorShape([])

  @property
  def output_types(self):
    return dtypes.string
