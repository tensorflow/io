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
"""GRPCEndpoint."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import concurrent.futures
import grpc
import google.protobuf.any_pb2

import tensorflow
# Incase test is done with TFIO_DATAPATH specified, the
# import path need to be extended to capture generated
# grpc files:
datapath = os.environ.get('TFIO_DATAPATH')
sys.path.append(os.path.abspath(
    os.path.dirname(__file__) if datapath is None else os.path.join(
        datapath, "tensorflow_io", "grpc", "python", "ops")))
import endpoint_pb2      # pylint: disable=wrong-import-position,unused-import
import endpoint_pb2_grpc # pylint: disable=wrong-import-position,unused-import

class GRPCEndpoint(endpoint_pb2_grpc.GRPCEndpointServicer):
  """GRPCEndpoint"""
  def __init__(self, data):
    self._grpc_server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=4))
    port = self._grpc_server.add_insecure_port("localhost:0")
    self._endpoint = "localhost:"+str(port)
    self._data = data
    super(GRPCEndpoint, self).__init__()
    endpoint_pb2_grpc.add_GRPCEndpointServicer_to_server(
        self, self._grpc_server)

  def start(self):
    self._grpc_server.start()

  def stop(self):
    self._grpc_server.stop(0)

  def endpoint(self):
    return self._endpoint

  def ReadRecord(self, request, context): # pylint: disable=unused-argument
    if len(self._data.shape) == 1:
      tensor = tensorflow.compat.v1.make_tensor_proto(
          self._data[request.offset:request.offset+request.length])
    else:
      tensor = tensorflow.compat.v1.make_tensor_proto(
          self._data[request.offset:request.offset+request.length, :])
    record = google.protobuf.any_pb2.Any()
    record.Pack(tensor)
    return endpoint_pb2.Response(record=record)
