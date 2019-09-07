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
"""GRPCIOServerClient."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import grpc
import google.protobuf.any_pb2

import tensorflow as tf

# In case test is done with TFIO_DATAPATH specified, the
# import path need to be extended to capture generated grpc files:
grpcpath = os.path.join(os.path.dirname(__file__), "..", "..")
datapath = os.environ.get('TFIO_DATAPATH')
if datapath is not None:
  grpcpath = os.path.join(datapath, "tensorflow_io", "grpc")
grpcpath = os.path.abspath(grpcpath)
sys.path.append(grpcpath)

import server_pb2      # pylint: disable=wrong-import-position
# For some reason generated code in grpc uses:
# from tensorflow_io.grpc import endpoint_pb2
# so below is needed
class MetaPathFinder(object):
  def find_module(self, fullname, _):
    if fullname == "tensorflow_io.grpc.server_pb2":
      return self
    return None
  def load_module(self, fullname):
    if fullname == "tensorflow_io.grpc.server_pb2":
      return server_pb2
    return None
sys.meta_path.append(MetaPathFinder())
import server_pb2_grpc # pylint: disable=wrong-import-position

class GRPCIOServerClient(object):
  """GRPCIOServerClient"""
  def __init__(self, server="localhost:50051"):
    self._server = server
    super(GRPCIOServerClient, self).__init__()

  def send(self, data, component):
    with grpc.insecure_channel(self._server) as channel:
      stub = server_pb2_grpc.GRPCIOServerStub(channel)
      metadata = [("component", component)]
      response = stub.Next(
          iter([server_pb2.Request(item=e) for e in data]),
          metadata=metadata)
      return response
