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

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


def read_fastq(filename, name=None):
    """Read FastQ file into Tensor

    Args:
        filename: Filename of the FastQ file.
        name: A name for the operation (optional).

    Returns:
        sequences: A string `Tensor`.
        raw_quality: A string `Tensor`.
    """
    return core_ops.io_read_fastq(filename, name=name)


@tf.function
def _nucleotide_to_onehot(nucleotide):
    """Encodes a nucleotide using a one hot encoding."""
    a_onehot = tf.constant([1, 0, 0, 0])
    c_onehot = tf.constant([0, 1, 0, 0])
    g_onehot = tf.constant([0, 0, 1, 0])
    t_onehot = tf.constant([0, 0, 0, 1])
    error_onehot = tf.constant([1, 1, 1, 1])

    if tf.math.equal(nucleotide, tf.constant(b"A")):  # pylint: disable=no-else-return
        return a_onehot
    elif tf.math.equal(nucleotide, tf.constant(b"C")):
        return c_onehot
    elif tf.math.equal(nucleotide, tf.constant(b"G")):
        return g_onehot
    elif tf.math.equal(nucleotide, tf.constant(b"T")):
        return t_onehot
    else:
        # TODO(suyashkumar): how best to raise error from within tf.function?
        return error_onehot


@tf.function
def sequences_to_onehot(sequences):
    """Convert DNA sequences into a one hot nucleotide encoding.

    Each nucleotide in each sequence is mapped as follows:
    A -> [1, 0, 0, 0]
    C -> [0, 1, 0, 0]
    G -> [0 ,0 ,1, 0]
    T -> [0, 0, 0, 1]

    If for some reason a non (A, T, C, G) character exists in the string, it is
    currently mapped to a error one hot encoding [1, 1, 1, 1].

    Args:
        sequences: A tf.string tensor where each string represents a DNA sequence

    Returns:
        tf.RaggedTensor: The output sequences with nucleotides one hot encoded.
    """
    all_onehot_nucleotides = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    sequence_splits = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    global_nucleotide_idx = 0
    sequence_splits = sequence_splits.write(
        sequence_splits.size(), global_nucleotide_idx
    )

    for sequence in sequences:
        for nucleotide in tf.strings.bytes_split(sequence):
            all_onehot_nucleotides = all_onehot_nucleotides.write(
                global_nucleotide_idx, _nucleotide_to_onehot(nucleotide)
            )
            global_nucleotide_idx += 1
        sequence_splits = sequence_splits.write(
            sequence_splits.size(), global_nucleotide_idx
        )
    return tf.RaggedTensor.from_row_splits(
        values=all_onehot_nucleotides.stack(),
        row_splits=tf.cast(sequence_splits.stack(), tf.int64),
    )


@tf.function
def _decode_byte_str(b_str):
    return tf.dtypes.cast(tf.strings.unicode_decode(b_str, "ASCII"), dtype=tf.float32)


@tf.function
def _phred_byte_to_probability(phred_byte_str):
    return tf.math.pow(10.0, -(_decode_byte_str(phred_byte_str) - 33) / 10)


@tf.function
def _phred_sequence_to_probability(seq_quality):
    return tf.map_fn(_phred_byte_to_probability, seq_quality, dtype=tf.float32)


@tf.function
def phred_sequences_to_probability(phred_qualities):
    """Converts raw phred quality scores into base-calling error probabilities.

    For each ASCII encoded phred quality score (X), the probability that there
    was an error calling that base is computed by:

    P = 10 ^ (-(X - 33) / 10)

    This is assuming an "ASCII base" of 33.

    The input is a tf.string tensor of ASCII encoded phred qualities,
    one string per DNA sequence, with each character representing the quality
    of a nucelotide.

    For example:
    phred_qualities = [["BB<"], ["BBBB"]]

    Args:
        phred_qualities: A tf.string tensor where each string represents the phred
                        quality of a DNA sequence. Each character in the string
                        is the ASCII representation of the phred quality number.

    Returns:
        tf.RaggedTensor: The quality scores for each base in each sequence provided.
    """
    return tf.ragged.map_flat_values(
        _phred_sequence_to_probability, tf.strings.bytes_split(phred_qualities)
    )
