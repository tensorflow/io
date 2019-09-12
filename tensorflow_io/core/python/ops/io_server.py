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
"""IOServer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class IOServer(object):
  """IOServer
  """
  def __init__(self,
               address="localhost:50001"):
    with tf.name_scope("IOServer") as scope:
      server = "GRPCIterable[%s]" % address
      metadata = ["server.address=%s" % address]
      resource = core_ops.grpcio_server_init(
          server, metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (server, uuid.uuid4().hex))

      self._resource = resource
      super(IOServer, self).__init__()
