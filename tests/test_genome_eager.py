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
import tensorflow_io as tfio # pylint: disable=wrong-import-position

fastq_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_genome", "test.fastq")

def test_genome_fastq_reader():
  """test_genome_fastq_reader"""

  data = tfio.genome.read_fastq(filename=fastq_path)

  data_expected = [
      b'GATTACA',
      b'CGTTAGCGCAGGGGGCATCTTCACACTGGTGACAGGTAACCGCCGTAGTAAAGGTTCCGCCTTTCACT',
      b'CGGCTGGTCAGGCTGACATCGCCGCCGGCCTGCAGCGAGCCGCTGC',
      b'CGG']

  quality_expected = [
      b'BB>B@FA',
      b'AAAAABF@BBBDGGGG?FFGFGHBFBFBFABBBHGGGFHHCEFGGGGG?FGFFHEDG3EFGGGHEGHG',
      b'FAFAF;F/9;.:/;999B/9A.DFFF;-->.AAB/FC;9-@-=;=.',
      b'FAD']

  assert np.all(data.sequences == data_expected)
  assert np.all(data.raw_quality == quality_expected)


def test_genome_sequences_to_onehot():
  """test sequence one hot encoder"""
  expected = [
      [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0],
       [0, 1, 0, 0], [1, 0, 0, 0]],
      [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0],
       [0, 0, 1, 0], [0, 1, 0, 0],
       [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
       [0, 0, 1, 0], [0, 0, 1, 0],
       [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0],
       [0, 0, 0, 1], [0, 0, 0, 1],
       [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
       [0, 0, 0, 1], [0, 0, 1, 0],
       [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0],
       [1, 0, 0, 0], [0, 0, 1, 0],
       [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
       [0, 1, 0, 0], [0, 0, 1, 0],
       [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0],
       [0, 0, 1, 0], [0, 0, 0, 1],
       [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
       [0, 0, 0, 1], [0, 0, 0, 1],
       [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0],
       [0, 0, 0, 1], [0, 0, 0, 1],
       [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
      [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1],
       [0, 0, 1, 0], [0, 0, 1, 0],
       [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
       [0, 1, 0, 0], [0, 0, 0, 1],
       [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1],
       [0, 1, 0, 0], [0, 0, 1, 0],
       [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0],
       [0, 0, 1, 0], [0, 0, 1, 0],
       [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
       [1, 0, 0, 0], [0, 0, 1, 0],
       [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
       [0, 1, 0, 0], [0, 0, 1, 0],
       [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
      [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]

  raw_data = tfio.genome.read_fastq(filename=fastq_path)
  data = tfio.genome.sequences_to_onehot(
      sequences=raw_data.sequences)

  assert np.all(data.to_list() == expected)


def test_genome_phred_sequences_to_probability():
  """Test conversion of phred qualities to probabilities"""
  example_quality_list = [b'BB<', b'ABFF']
  expected_probabilities = [0.0005011872854083776, 0.0005011872854083776,
                            0.0019952619913965464, 0.0006309572490863502,
                            0.0005011872854083776, 0.00019952621369156986,
                            0.00019952621369156986]

  example_quality = tf.constant(example_quality_list)
  converted_phred = tfio.genome.phred_sequences_to_probability(
      example_quality)

  # Compare flat values
  assert np.allclose(
      converted_phred.flat_values.numpy().flatten(), expected_probabilities)
  # Ensure nested array lengths are correct
  assert np.all(
      [len(a) == len(b)
       for a, b in zip(converted_phred.to_list(), example_quality_list)])

if __name__ == "__main__":
  test.main()
