"""LibSVM Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.experimental.ops import batching

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

libsvm_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_libsvm_ops.so'))


decode_libsvm = libsvm_ops.decode_libsvm


def make_libsvm_dataset(file_names,
                        num_features,
                        dtype=None,
                        label_dtype=None,
                        batch_size=1,
                        compression_type='',
                        buffer_size=None,
                        num_parallel_parser_calls=None,
                        drop_final_batch=False,
                        prefetch_buffer_size=0):
    """Reads LibSVM files into a dataset.

  Args:
    file_names: A `tf.string` tensor containing one or more filenames.
    num_features: The number of features.
    dtype(Optional): The type of the output feature tensor. Default to tf.float32.
    label_dtype(Optional): The type of the output label tensor. Default to tf.int64.
    batch_size: (Optional.) An int representing the number of records to combine
      in a single batch, default 1.
    compression_type: (Optional.) A `tf.string` scalar evaluating to one of
      `""` (no compression), `"ZLIB"`, or `"GZIP"`.
    buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
      to buffer. A value of 0 results in the default buffering values chosen
      based on the compression type.
    num_parallel_parser_calls: (Optional.) Number of parallel
      records to parse in parallel. Defaults to an automatic selection.
    drop_final_batch: (Optional.) Whether the last batch should be
      dropped in case its size is smaller than `batch_size`; the
      default behavior is not to drop the smaller batch.
    prefetch_buffer_size: (Optional.) An int specifying the number of
      feature batches to prefetch for performance improvement.
      Defaults to auto-tune. Set to 0 to disable prefetching.
  """
  dataset = core_readers.TextLineDataset(file_names, 
                                         compression_type=compression_type, 
                                         buffer_size=buffer_size)
  parsing_func = partial(decode_libsvm,
            num_features=num_features, 
            dtype=dtype, 
            label_type=label_type)
  dataset = dataset.apply(batching.map_and_batch(
                                        parsing_func, 
                                        batch_size, 
                                        num_parallel_calls=num_parallel_parser_calls,
                                        drop_remainder=drop_final_batch))
    if prefetch_buffer_size == 0:
        return dataset
    else:
        return dataset.prefetch(buffer_size=prefetch_buffer_size)
