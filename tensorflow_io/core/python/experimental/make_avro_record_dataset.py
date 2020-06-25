# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""make_avro_record_dataset"""

import tensorflow as tf

from tensorflow_io.core.python.experimental.avro_record_dataset_ops import (
    AvroRecordDataset,
)
from tensorflow_io.core.python.experimental.parse_avro_ops import parse_avro

# heavily inspired by make_tf_record_dataset from
# https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/data/experimental/ops/readers.py


def make_avro_record_dataset(
    file_pattern,
    features,
    batch_size,
    reader_schema,
    reader_buffer_size=None,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=None,
    shuffle_seed=None,
    prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
    num_parallel_reads=None,
    num_parallel_parser_calls=None,
    drop_final_batch=False,
):
    """Reads and (optionally) parses avro files into a dataset.

    Provides common functionality such as batching, optional parsing, shuffling,
    and performing defaults.

    Args:
      file_pattern: List of files or patterns of avro file paths.
        See `tf.io.gfile.glob` for pattern rules.

      features: A map of feature names mapped to feature information.

      batch_size: An int representing the number of records to combine
        in a single batch.

      reader_schema: The reader schema.

      reader_buffer_size: (Optional.) An int specifying the readers buffer
        size in By. If None (the default) will use the default value from
        AvroRecordDataset.

      num_epochs: (Optional.) An int specifying the number of times this
        dataset is repeated.  If None (the default), cycles through the
        dataset forever. If set to None drops final batch.

      shuffle: (Optional.) A bool that indicates whether the input
        should be shuffled. Defaults to `True`.

      shuffle_buffer_size: (Optional.) Buffer size to use for
        shuffling. A large buffer size ensures better shuffling, but
        increases memory usage and startup time. If not provided
        assumes default value of 10,000 records. Note that the shuffle
        size is measured in records.

      shuffle_seed: (Optional.) Randomization seed to use for shuffling.
        By default uses a pseudo-random seed.

      prefetch_buffer_size: (Optional.) An int specifying the number of
        feature batches to prefetch for performance improvement.
        Defaults to auto-tune. Set to 0 to disable prefetching.

      num_parallel_reads: (Optional.) Number of threads used to read
        records from files. By default or if set to a value >1, the
        results will be interleaved.

      num_parallel_parser_calls: (Optional.) Number of parallel
        records to parse in parallel. Defaults to an automatic selection.

      drop_final_batch: (Optional.) Whether the last batch should be
        dropped in case its size is smaller than `batch_size`; the
        default behavior is not to drop the smaller batch.

    Returns:
      A dataset, where each element matches the output of `parser_fn`
      except it will have an additional leading `batch-size` dimension,
      or a `batch_size`-length 1-D tensor of strings if `parser_fn` is
      unspecified.
    """
    files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle, seed=shuffle_seed)

    if num_parallel_reads is None:
        # Note: We considered auto-tuning this value, but there is a concern
        # that this affects the mixing of records from different files, which
        # could affect training convergence/accuracy, so we are defaulting to
        # a constant for now.
        num_parallel_reads = 24

    if reader_buffer_size is None:
        reader_buffer_size = 1024 * 1024

    dataset = AvroRecordDataset(
        files,
        buffer_size=reader_buffer_size,
        num_parallel_reads=num_parallel_reads,
        reader_schema=reader_schema,
    )

    if shuffle_buffer_size is None:
        # TODO(josh11b): Auto-tune this value when not specified
        shuffle_buffer_size = 10000
    dataset = _maybe_shuffle_and_repeat(
        dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed
    )

    # NOTE(mrry): We set `drop_final_batch=True` when `num_epochs is None` to
    # improve the shape inference, because it makes the batch dimension static.
    # It is safe to do this because in that case we are repeating the input
    # indefinitely, and all batches will be full-sized.
    drop_final_batch = drop_final_batch or num_epochs is None

    dataset = dataset.batch(batch_size, drop_remainder=drop_final_batch)

    if num_parallel_parser_calls is None:
        num_parallel_parser_calls = tf.data.experimental.AUTOTUNE

    dataset = dataset.map(
        lambda data: parse_avro(
            serialized=data, reader_schema=reader_schema, features=features
        ),
        num_parallel_calls=num_parallel_parser_calls,
    )

    if prefetch_buffer_size == 0:
        return dataset
    return dataset.prefetch(buffer_size=prefetch_buffer_size)


def _maybe_shuffle_and_repeat(
    dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed
):
    """Optionally shuffle and repeat dataset, as requested."""
    if num_epochs != 1 and shuffle:
        # Use shuffle_and_repeat for perf
        return dataset.apply(
            tf.data.experimental.shuffle_and_repeat(
                shuffle_buffer_size, num_epochs, shuffle_seed
            )
        )
    if shuffle:
        return dataset.shuffle(shuffle_buffer_size, shuffle_seed)
    if num_epochs != 1:
        return dataset.repeat(num_epochs)
    return dataset
