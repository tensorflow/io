# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Serialization Ops."""

import json

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

# _NamedTensorSpec allows adding a `named` key while traversing,
# so that it is possible to build up the `/R/Foo` JSON Pointers.
class _NamedTensorSpec(tf.TensorSpec):
    """_NamedTensorSpec"""

    def named(self, named=None):
        if named is not None:
            self._named = named
        return self._named


# named_spec updates named field for JSON Pointers while traversing.
def named_spec(specs, name=""):
    """named_spec"""
    if isinstance(specs, _NamedTensorSpec):
        specs.named(name)
        return

    if isinstance(specs, dict):
        for k in specs.keys():
            named_spec(specs[k], "{}/{}".format(name, k))
        return

    for k, _ in enumerate(specs):
        named_spec(specs[k], "{}/{}".format(name, k))
    return


def decode_json(data, specs, name=None):
    """
    Decode JSON string into Tensors.

    Args:
        data: A String Tensor. The JSON strings to decode.
        specs: A structured TensorSpecs describing the signature
        of the JSON elements.
        name: A name for the operation (optional).

    Returns:
        A structured Tensors.
    """
    # TODO: support batch (1-D) input
    # Make a copy of specs to keep the original specs
    named = tf.nest.map_structure(lambda e: _NamedTensorSpec(e.shape, e.dtype), specs)
    named_spec(named)
    named = tf.nest.flatten(named)
    names = [e.named() for e in named]
    shapes = [
        tf.constant([-1 if d is None else d for d in e.shape.as_list()], tf.int32)
        for e in named
    ]
    dtypes = [e.dtype for e in named]

    values = core_ops.io_decode_json(data, names, dtypes, name=name)
    values = [tf.reshape(value, shape) for value, shape in zip(values, shapes)]
    return tf.nest.pack_sequence_as(specs, values)


def process_primitive(data, name):
    """process_primitive"""
    if data == "boolean":
        return tf.TensorSpec(tf.TensorShape([]), tf.bool, name)
    if data == "int":
        return tf.TensorSpec(tf.TensorShape([]), tf.int32, name)
    if data == "long":
        return tf.TensorSpec(tf.TensorShape([]), tf.int64, name)
    if data == "float":
        return tf.TensorSpec(tf.TensorShape([]), tf.float32, name)
    if data == "double":
        return tf.TensorSpec(tf.TensorShape([]), tf.float64, name)
    assert data in ("bytes", "string")
    return tf.TensorSpec(tf.TensorShape([]), tf.string, name)


def process_record(data, name):
    """process_record"""
    return {
        v["name"]: process_entry(v, "{}/{}".format(name, v["name"]))
        for v in data["fields"]
    }


def process_union(data, name):
    """process_union"""
    entries = [e for e in data["type"] if e != "null"]
    assert len(entries) == 1
    return process_primitive(entries[0], name)


def process_entry(data, name):
    """process_entry"""
    if data["type"] == "record":
        return process_record(data, name)
    if data["type"] == "enum":
        assert False
    if data["type"] == "array":
        assert False
    if data["type"] == "map":
        assert False
    if data["type"] == "fixed":
        assert False
    if isinstance(data["type"], list):
        return process_union(data, name)
    return process_primitive(data["type"], name)


def decode_avro(data, schema, name=None):
    """
    Decode Avro string into Tensors.

    Args:
        data: A String Tensor. The Avro strings to decode.
        schema: A string of the Avro schema.
        name: A name for the operation (optional).

    Returns:
        A structured Tensors.
    """
    # TODO: Use resource to reuse schema initialization
    specs = process_entry(
        json.loads(schema.decode() if isinstance(schema, bytes) else schema), ""
    )

    entries = tf.nest.flatten(specs)
    names = [e.name for e in entries]
    shapes = [e.shape for e in entries]
    dtypes = [e.dtype for e in entries]

    values = core_ops.io_decode_avro(data, names, schema, shapes, dtypes, name=name)
    return tf.nest.pack_sequence_as(specs, values)


def encode_avro(data, schema, name=None):
    """
    Encode Tensors into Avro string.

    Args:
        data: A list of Tensors to encode.
        schema: A string of the Avro schema.
        name: A name for the operation (optional).

    Returns:
        An Avro-encoded string Tensor.
    """
    # TODO: Use resource to reuse schema initialization
    specs = process_entry(
        json.loads(schema.decode() if isinstance(schema, bytes) else schema), ""
    )

    entries = tf.nest.flatten(specs)
    names = [e.name for e in entries]

    data = tf.nest.flatten(data)

    values = core_ops.io_encode_avro(data, names, schema, name=name)
    return values
