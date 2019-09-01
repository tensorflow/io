from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import tensorflow_io.genome as genome_io
dir(genome_io)

def test_genome_fastq_reader():
    g1 = tf.compat.v1.Graph()

    with g1.as_default():
        data = genome_io.fastq_op(filename="test.fastq")

    sess = tf.compat.v1.Session(graph=g1)
    data_np = sess.run(data)

if __name__ == "__main__":
    test.main()
