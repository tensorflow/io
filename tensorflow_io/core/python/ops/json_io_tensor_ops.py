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
"""JSONIOTensor"""

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops


class JSONIOTensor(io_tensor_ops._TableIOTensor):  # pylint: disable=protected-access
    """JSONIOTensor"""

    # =============================================================================
    # Constructor (private)
    # =============================================================================
    def __init__(self, filename, mode=None, internal=False):
        with tf.name_scope("JSONIOTensor") as scope:
            metadata = [] if mode is None else ["mode: %s" % mode]
            resource, columns = core_ops.io_json_readable_init(
                filename,
                metadata=metadata,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )
            columns = [column.decode() for column in columns.numpy().tolist()]
            elements = []
            for column in columns:
                shape, dtype = core_ops.io_json_readable_spec(resource, column)
                shape = tf.TensorShape(shape.numpy())
                dtype = tf.as_dtype(dtype.numpy())
                spec = tf.TensorSpec(shape, dtype, column)
                function = io_tensor_ops._IOTensorComponentFunction(  # pylint: disable=protected-access
                    core_ops.io_json_readable_read, resource, column, shape, dtype
                )
                elements.append(
                    io_tensor_ops.BaseIOTensor(spec, function, internal=internal)
                )
            spec = tuple([e.spec for e in elements])
            super().__init__(spec, columns, elements, internal=internal)
