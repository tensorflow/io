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
"""Genome"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

read_fastq = core_ops.read_fastq

A_ONEHOT = tf.constant([1, 0, 0, 0])
C_ONEHOT = tf.constant([0, 1, 0, 0])
G_ONEHOT = tf.constant([0, 0, 1, 0])
T_ONEHOT = tf.constant([0, 0, 0, 1])
ERROR = tf.constant([1, 1, 1, 1])


@tf.function
def _nucleotide_to_onehot(nucleotide):
    if tf.math.equal(nucleotide, tf.constant(b'A')):
        return A_ONEHOT
    elif tf.math.equal(nucleotide, tf.constant(b'C')):
        return C_ONEHOT
    elif tf.math.equal(nucleotide, tf.constant(b'G')):
        return G_ONEHOT
    elif tf.math.equal(nucleotide, tf.constant(b'T')):
        return T_ONEHOT
    else:
        # TODO(suyashkumar): how best to raise error from within tf.function?
        return ERROR


@tf.function
def sequences_to_onehot(sequences):
    sequences_onehot = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    splits = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    i = 0
    splits = splits.write(splits.size(), i)
    for sequence in sequences:
        split_nucleotides = tf.strings.bytes_split(sequence)
        for nucleotide in split_nucleotides:
            sequences_onehot = sequences_onehot.write(sequences_onehot.size(), _nucleotide_to_onehot(nucleotide))
            i = i + 1
        splits = splits.write(splits.size(), i)
    return tf.RaggedTensor.from_row_splits(values=sequences_onehot.stack(), row_splits=splits.stack())
