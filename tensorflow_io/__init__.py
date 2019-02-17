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
"""tensorflow-io"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import inspect

from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

# NOTE: __datapath__ is a hidden value used by test,
# in case .so files are on a different path.
__datapath__ = None

def _load_library(filename, lib="op"):
  f = inspect.getfile(sys._getframe(1))
  # Construct filename
  f = os.path.join(os.path.dirname(f), filename)
  filenames = [f];
  if __datapath__ is not None:
    f = os.path.join(__datapath__, __name__, os.path.relpath(f, os.path.dirname(__file__)))
    filenames.append(f);
  fn = load_library.load_op_library if lib == "op" else load_library.load_file_system_library
  for f in filenames:
    try:
      if lib == "op":
        l = load_library.load_op_library(f)
      else:
        load_library.load_file_system_library(f)
        l = True
      if l is not None:
        return l
    except errors_impl.NotFoundError as e:
      pass
  raise NotImplementedError("unable to open file: ", filename)
