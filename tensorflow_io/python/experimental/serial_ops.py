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
"""Easily save tf.data.Datasets as tfrecord files, and restore tfrecords as Datasets.

The goal of this module is to create a SIMPLE api to tfrecords that can be used without
learning all of the underlying mechanics.

Users only need to deal with 2 functions:
save_dataset(dataset)
dataset = load_dataset(tfrecord, header)

It really is that easy!

To make this work, we create a .header file for each tfrecord which encodes metadata
needed to reconstruct the original dataset.

Note that PyYAML (yaml) package must be installed to make use of this module.

Saving must be done in eager mode, but loading is compatible with both eager and
graph execution modes.

GOTCHAS:
- This module is only compatible with "dictionary-style" datasets {key: val, key2:val2,..., keyN: valN}.
- The restored dataset will have the TFRecord dtypes {float32, int64, string} instead of the original
 tensor dtypes. This is always the case with TFRecord datasets, whether you use this module or not.
 The original dtypes are stored in the headers if you want to restore them after loading."""
import functools
import os
import tempfile

import numpy as np
import tensorflow as tf


# The three encoding functions.
def _bytes_feature(value):
    """value: list"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """value: list"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """value: list"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# TODO use base_type() to ensure consistent conversion.
def np_value_to_feature(value):
    """Maps dataset values to tf Features.
    Only numpy types are supported since Datasets only contain tensors.
    Each datatype should only have one way of being serialized."""
    if isinstance(value, np.ndarray):
        # feature = _bytes_feature(value.tostring())
        if np.issubdtype(value.dtype, np.integer):
            feature = _int64_feature(value.flatten())
        elif np.issubdtype(value.dtype, np.float):
            feature = _float_feature(value.flatten())
        elif np.issubdtype(value.dtype, np.bool):
            feature = _int64_feature(value.flatten())
        else:
            raise TypeError(f"value dtype: {value.dtype} is not recognized.")
    elif isinstance(value, bytes):
        feature = _bytes_feature([value])
    elif np.issubdtype(type(value), np.integer):
        feature = _int64_feature([value])
    elif np.issubdtype(type(value), np.float):
        feature = _float_feature([value])

    else:
        raise TypeError(
            f"value type: {type(value)} is not recognized. value must be a valid Numpy object."
        )

    return feature


def base_type(dtype):
    """Returns the TFRecords allowed type corresponding to dtype."""
    int_types = [
        tf.int8,
        tf.int16,
        tf.int32,
        tf.int64,
        tf.uint8,
        tf.uint16,
        tf.uint32,
        tf.uint64,
        tf.qint8,
        tf.qint16,
        tf.qint32,
        tf.bool,
    ]
    float_types = [tf.float16, tf.float32, tf.float64]
    byte_types = [tf.string, bytes]

    if dtype in int_types:
        new_dtype = tf.int64
    elif dtype in float_types:
        new_dtype = tf.float32
    elif dtype in byte_types:
        new_dtype = tf.string
    else:
        raise ValueError(f"dtype {dtype} is not a recognized/supported type!")

    return new_dtype


def build_header(dataset):
    """Build header dictionary of metadata for the tensors in the dataset. This will be used when loading
    the tfrecords file to reconstruct the original tensors from the raw data. Shape is stored as an array
    and dtype is stored as an enumerated value (defined by tensorflow)."""
    header = {}
    for key in dataset.element_spec.keys():
        header[key] = {
            "shape": list(dataset.element_spec[key].shape),
            "dtype": dataset.element_spec[key].dtype.as_datatype_enum,
        }

    return header


def build_feature_desc(header):
    """Build feature_desc dictionary for the tensors in the dataset. This will be used to reconstruct Examples
    from the tfrecords file.

    Assumes FixedLenFeatures.
    If you got VarLenFeatures I feel bad for you son,
    I got 115 problems but a VarLenFeature ain't one."""
    feature_desc = {}
    for key, params in header.items():
        feature_desc[key] = tf.io.FixedLenFeature(
            shape=params["shape"], dtype=base_type(int(params["dtype"]))
        )

    return feature_desc


def dataset_to_examples(ds):
    """Converts a dataset to a dataset of tf.train.Example strings. Each Example is a single observation.
    WARNING: Only compatible with "dictionary-style" datasets {key: val, key2:val2,..., keyN, valN}.
    WARNING: Must run in eager mode!"""
    # TODO handle tuples and flat datasets as well.
    for x in ds:
        # Each individual tensor is converted to a known serializable type.
        features = {key: np_value_to_feature(value.numpy()) for key, value in x.items()}
        # All features are then packaged into a single Example object.
        example = tf.train.Example(features=tf.train.Features(feature=features))

        yield example.SerializeToString()


def save_dataset(dataset, tfrecord_path, header_path):
    """Saves a flat dataset as a tfrecord file, and builds a header file for reloading as dataset.
    Must run in eager mode because it depends on dataset iteration and element_spec."""
    import yaml

    if not tf.executing_eagerly():
        raise ValueError("save_dataset() must run in eager mode!")

    # Header
    header = build_header(dataset)
    header_file = open(header_path, "w")
    yaml.dump(header, stream=header_file)

    # Dataset
    ds_examples = tf.data.Dataset.from_generator(
        lambda: dataset_to_examples(dataset), output_types=tf.string
    )
    writer = tf.data.experimental.TFRecordWriter(tfrecord_path)
    writer.write(ds_examples)


# TODO-DECIDE is this yaml loader safe?
def load_dataset(tfrecord_path, header_path):
    """Uses header file to predict the shape and dtypes of tensors for tf.data."""
    import yaml

    header_file = open(header_path)
    header = yaml.load(header_file, Loader=yaml.FullLoader)

    feature_desc = build_feature_desc(header)
    parse_func = functools.partial(tf.io.parse_single_example, features=feature_desc)
    dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_func)

    return dataset
