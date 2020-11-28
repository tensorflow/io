# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import ctypes
import _ctypes

from tensorflow_io.core.python.ops import _load_library

_mongodb_ops = _load_library("libtensorflow_io_mongodb.so")

io_mongo_db_readable_init = _mongodb_ops.io_mongo_db_readable_init
io_mongo_db_readable_next = _mongodb_ops.io_mongo_db_readable_next
io_mongo_db_writable_init = _mongodb_ops.io_mongo_db_writable_init
io_mongo_db_writable_write = _mongodb_ops.io_mongo_db_writable_write
io_mongo_db_writable_delete_many = _mongodb_ops.io_mongo_db_writable_delete_many
