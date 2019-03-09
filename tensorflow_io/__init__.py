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

import tensorflow
from tensorflow import errors

def _load_library(filename, lib="op"):
  """_load_library"""
  f = inspect.getfile(sys._getframe(1)) # pylint: disable=protected-access

  # Construct filename
  f = os.path.join(os.path.dirname(f), filename)
  filenames = [f]

  # Add datapath to load if en var is set, used for running tests where shared
  # libraries are built in a different path
  datapath = os.environ.get('TFIO_DATAPATH')
  if datapath is not None:
    # Build filename from `datapath` + `package_name` + `relpath_to_library`
    f = os.path.join(
        datapath, __name__, os.path.relpath(f, os.path.dirname(__file__)))
    filenames.append(f)

  # Function to load the library, return True if file system library is loaded
  load_fn = tensorflow.load_op_library if lib == "op" \
      else lambda f: tensorflow.compat.v1.load_file_system_library(f) is None

  # Try to load all paths for file, fail if none succeed
  errs = []
  for f in filenames:
    try:
      l = load_fn(f)
      if l is not None:
        return l
    except errors.NotFoundError as e:
      errs.append(str(e))
  raise NotImplementedError(
      "unable to open file: " +
      "{}, from paths: {}\ncaused by: {}".format(filename, filenames, errs))
