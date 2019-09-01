from __future__ import absolute_import
from tensorflow_io.core.python.ops import _load_library
genome_ops = _load_library('_genome_ops.so')
dir(genome_ops)

fastq_op = genome_ops.fastq_op
