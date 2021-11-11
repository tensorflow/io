# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""GS"""

import os
import ctypes
import sys
import inspect
import warnings
import types

import tensorflow as tf


def _load_library(filename):
    """_load_library"""
    f = inspect.getfile(sys._getframe(1))  # pylint: disable=protected-access

    # Construct filename
    f = os.path.join(os.path.dirname(f), filename)
    filenames = [f]

    # Add datapath to load if en var is set, used for running tests where shared
    # libraries are built in a different path
    datapath = os.environ.get("TFIO_DATAPATH")
    if datapath is not None:
        # Build filename from:
        # `datapath` + `tensorflow_io_package` + `package_name` + `relpath_to_library`
        rootpath = os.path.dirname(sys.modules["tensorflow_io_gcs_filesystem"].__file__)
        filename = sys.modules[__name__].__file__
        f = os.path.join(
            datapath,
            "tensorflow_io_gcs_filesystem",
            os.path.relpath(os.path.dirname(filename), rootpath),
            os.path.relpath(f, os.path.dirname(filename)),
        )
        filenames.append(f)
    # Function to load the library, return True if file system library is loaded
    load_fn = lambda f: tf.experimental.register_filesystem_plugin(f) is None

    # Try to load all paths for file, fail if none succeed
    errs = []
    for f in filenames:
        try:
            l = load_fn(f)
            if l is not None:
                return l
        except (tf.errors.NotFoundError, OSError) as e:
            errs.append(str(e))
    raise NotImplementedError(
        "unable to open file: "
        + f"{filename}, from paths: {filenames}\ncaused by: {errs}"
    )


try:
    plugin_gs = _load_library("libtensorflow_io_gcs_filesystem.so")
except NotImplementedError as e:
    warnings.warn(f"file system plugin for gs are not loaded: {e}")
