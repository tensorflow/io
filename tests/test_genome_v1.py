# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for Genome."""


import os
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio


def test_genome_fastq_reader():
    """test_genome_fastq_reader"""
    tf.compat.v1.disable_eager_execution()
    fastq_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_genome", "test.fastq"
    )
    g1 = tf.compat.v1.Graph()

    with g1.as_default():
        data = tfio.genome.read_fastq(filename=fastq_path)

    sess = tf.compat.v1.Session(graph=g1)
    data_np = sess.run(data)

    data_expected = [
        b"GATTACA",
        b"CGTTAGCGCAGGGGGCATCTTCACACTGGTGACAGGTAACCGCCGTAGTAAAGGTTCCGCCTTTCACT",
        b"CGGCTGGTCAGGCTGACATCGCCGCCGGCCTGCAGCGAGCCGCTGC",
        b"CGG",
    ]

    quality_expected = [
        b"BB>B@FA",
        b"AAAAABF@BBBDGGGG?FFGFGHBFBFBFABBBHGGGFHHCEFGGGGG?FGFFHEDG3EFGGGHEGHG",
        b"FAFAF;F/9;.:/;999B/9A.DFFF;-->.AAB/FC;9-@-=;=.",
        b"FAD",
    ]

    assert np.all(data_np.sequences == data_expected)
    assert np.all(data_np.raw_quality == quality_expected)


def test_genome_sequences_to_onehot():
    """test sequence one hot encoder"""
    tf.compat.v1.disable_eager_execution()
    fastq_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_genome", "test.fastq"
    )
    expected = [
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    ]

    with tf.compat.v1.Session() as sess:
        raw_data = tfio.genome.read_fastq(filename=fastq_path)
        data = tfio.genome.sequences_to_onehot(sequences=raw_data.sequences)
        out = sess.run(data)

    assert np.all(out.to_list() == expected)


def test_genome_phred_sequences_to_probability():
    """Test conversion of phred qualities to probabilities"""
    tf.compat.v1.disable_eager_execution()
    fastq_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_genome", "test.fastq"
    )
    example_quality_list = [b"BB<", b"ABFF"]
    expected_probabilities = [
        0.0005011872854083776,
        0.0005011872854083776,
        0.0019952619913965464,
        0.0006309572490863502,
        0.0005011872854083776,
        0.00019952621369156986,
        0.00019952621369156986,
    ]

    with tf.compat.v1.Session() as sess:
        example_quality = tf.constant(example_quality_list)
        converted_phred = tfio.genome.phred_sequences_to_probability(example_quality)
        out = sess.run(converted_phred)

    # Compare flat values
    assert np.allclose(out.flat_values.flatten(), expected_probabilities)
    # Ensure nested array lengths are correct
    assert np.all(
        [len(a) == len(b) for a, b in zip(out.to_list(), example_quality_list)]
    )


def test_genome_phred_sequences_to_probability_with_other_genome_ops():
    """Test quality op in graph with read_fastq op, ensure no errors"""
    tf.compat.v1.disable_eager_execution()
    fastq_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_genome", "test.fastq"
    )
    with tf.compat.v1.Session() as sess:
        raw_data = tfio.genome.read_fastq(filename=fastq_path)
        data = tfio.genome.phred_sequences_to_probability(
            phred_qualities=raw_data.raw_quality
        )
        sess.run(data)


if __name__ == "__main__":
    test.main()
