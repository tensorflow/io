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
"""Archive."""

from tensorflow_io.core.python.ops import core_ops


def list_archive_entries(filename, filters, **kwargs):
    """list_archive_entries"""
    memory = kwargs.get("memory", "")
    if not isinstance(filters, list):
        filters = [filters]
    return core_ops.io_list_archive_entries(filename, filters=filters, memory=memory)


def read_archive(
    filename, format, entries, **kwargs
):  # pylint: disable=redefined-builtin
    """read_archive"""
    memory = kwargs.get("memory", "")
    return core_ops.io_read_archive(filename, format, entries, memory=memory)
