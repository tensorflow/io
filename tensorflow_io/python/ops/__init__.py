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
"""Dataset."""

import os
import ctypes
import sys
import inspect
import warnings
import types

import tensorflow as tf


def _load_library(filename, lib="op"):
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
        # `datapath` + `tensorflow_io` + `package_name` + `relpath_to_library`
        rootpath = os.path.dirname(sys.modules["tensorflow_io"].__file__)
        filename = sys.modules[__name__].__file__
        f = os.path.join(
            datapath,
            "tensorflow_io",
            os.path.relpath(os.path.dirname(filename), rootpath),
            os.path.relpath(f, os.path.dirname(filename)),
        )
        filenames.append(f)
    # Function to load the library, return True if file system library is loaded
    if lib == "op":
        load_fn = tf.load_op_library
    elif lib == "dependency":
        load_fn = lambda f: ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
    elif lib == "fs":
        load_fn = lambda f: tf.experimental.register_filesystem_plugin(f) is None
    else:
        load_fn = lambda f: tf.compat.v1.load_file_system_library(f) is None

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
        + "{}, from paths: {}\ncaused by: {}".format(filename, filenames, errs)
    )


class LazyLoader(types.ModuleType):
    def __init__(self, name, library):
        self._mod = None
        self._module_name = name
        self._library = library
        super().__init__(self._module_name)

    def _load(self):
        if self._mod is None:
            self._mod = _load_library(self._library)
        return self._mod

    def __getattr__(self, attrb):
        return getattr(self._load(), attrb)

    def __dir__(self):
        return dir(self._load())


core_ops = LazyLoader("core_ops", "libtensorflow_io.so")
try:
    plugin_ops = _load_library("libtensorflow_io_plugins.so", "fs")
except NotImplementedError as e:
    warnings.warn("unable to load libtensorflow_io_plugins.so: {}".format(e))
    # Note: load libtensorflow_io.so imperatively in case of statically linking
    try:
        core_ops = _load_library("libtensorflow_io.so")
        plugin_ops = _load_library("libtensorflow_io.so", "fs")
    except NotImplementedError as e:
        warnings.warn("file system plugins are not loaded: {}".format(e))
