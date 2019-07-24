from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import perf_counter
import logging
import glob
import os

import tensorflow as tf
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.ops import parsing_ops

import tensorflow_io.avro.python.ops.avro_dataset as avro_io


if __name__ == "__main__":

  batch_size = 512
  num_parallel_calls = 1
  num_parallel_reads = 1

  filepath = "/home/fraudies/Desktop/tmph1epww3x"
  filenames = glob.glob(os.path.join(filepath, "*.avro"))
  reader_schema_str = """
  {
    "type": "record",
    "name": "person",
    "fields": [
        {"name": "employed", "type": "boolean", "optional": true, "default": false},
        {"name": "age", "type": "int"},
        {"name": "id", "type": "long"},
        {"name": "salary", "type": "float"},
        {"name": "altitude", "type": "double"},
        {"name": "name", "type": "string"}
   ]
  }"""
  features = {
    "employed": parsing_ops.FixedLenFeature([], tf_types.bool),
    "age": parsing_ops.FixedLenFeature([], tf_types.int32),
    "id": parsing_ops.FixedLenFeature([], tf_types.int64),
    "salary": parsing_ops.FixedLenFeature([], tf_types.float32),
    "altitude": parsing_ops.FixedLenFeature([], tf_types.float64),
    "name": parsing_ops.FixedLenFeature([], tf_types.string)
  }
  shuffle = False
  num_epochs = 10

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # All > 1 are logged
  # os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '100'  # All < 10 are logged

  # Only use a 40% of the available GPU memory when testing
  # Need to keep some memory for the graphics etc.
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  config.intra_op_parallelism_threads = 1
  config.inter_op_parallelism_threads = 1
  tf.enable_eager_execution(config=config)

  logging.info("TensorFlow version: {}".format(tf.__version__))
  logging.info("Eager execution: {}".format(tf.executing_eagerly()))

  dataset = avro_io.make_avro_dataset(
      file_pattern=filenames,
      reader_schema=reader_schema_str,
      features=features,
      num_parallel_calls=num_parallel_calls,
      batch_size=batch_size,
      shuffle=shuffle,
      num_epochs=num_epochs,
      num_parallel_reads=num_parallel_reads)

  num_samples = 0
  start_time = perf_counter()
  for datum in iter(dataset):
    logging.info("datum {}".format(datum))
    num_samples += batch_size
  duration = perf_counter() - start_time

  print("It took {}", duration)
