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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_io.genome as genome_io # pylint: disable=wrong-import-position

fastq_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_genome", "test.fastq")

def test_genome_fastq_reader():
  """test_genome_fastq_reader"""
  g1 = tf.compat.v1.Graph()

  with g1.as_default():
    data = genome_io.read_fastq(filename=fastq_path)

  sess = tf.compat.v1.Session(graph=g1)
  data_np = sess.run(data)

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

  assert np.all(data_np.sequences == data_expected)
  assert np.all(data_np.raw_quality == quality_expected)


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

  with tf.compat.v1.Session() as sess:
    raw_data = genome_io.read_fastq(filename=fastq_path)
    data = genome_io.sequences_to_onehot(sequences=raw_data.sequences)
    out = sess.run(data)

  assert np.all(out.to_list() == expected)

if __name__ == "__main__":
  test.main()
