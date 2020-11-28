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
"""Module to wrap the mongoDB ops"""
import json


def readable_init(uri, database, collection):

    from tensorflow_io.core.python.ops import (
        mongodb_ops,
    )  # pylint: disable=import-outside-toplevel

    return mongodb_ops.io_mongo_db_readable_init(
        uri=uri, database=database, collection=collection,
    )


def readable_next(resource):

    from tensorflow_io.core.python.ops import (
        mongodb_ops,
    )  # pylint: disable=import-outside-toplevel

    return mongodb_ops.io_mongo_db_readable_next(resource=resource)


def writable_init(uri, database, collection):

    from tensorflow_io.core.python.ops import (
        mongodb_ops,
    )  # pylint: disable=import-outside-toplevel

    return mongodb_ops.io_mongo_db_writable_init(
        uri=uri, database=database, collection=collection,
    )


def writable_write(resource, doc):

    from tensorflow_io.core.python.ops import (
        mongodb_ops,
    )  # pylint: disable=import-outside-toplevel

    mongodb_ops.io_mongo_db_writable_write(resource=resource, record=json.dumps(doc))


def _delete_many(resource, doc):

    from tensorflow_io.core.python.ops import (
        mongodb_ops,
    )  # pylint: disable=import-outside-toplevel

    mongodb_ops.io_mongo_db_writable_delete_many(
        resource=resource, record=json.dumps(doc)
    )
