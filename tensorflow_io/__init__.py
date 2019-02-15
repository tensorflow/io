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

def _load_library(filename):
  f = inspect.getfile(sys._getframe(1))
  # Construct filename
  f = os.path.join(os.path.dirname(f), filename)
  filenames = [f];
  if not os.path.isabs(f):
    filenames.extend([os.path.join(p, f) for p in sys.path])
  for f in filenames:
    try:
      l = load_library.load_op_library(f)
      if l is not None:
        return l
    except errors_impl.NotFoundError as e:
      pass
  raise NotImplementedError("unable to open file: ", filename)
