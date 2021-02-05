import tests.test_benchmark.benchmark_dataset as benchmark_dataset

from tensorflow_io.core.python.experimental.make_avro_record_dataset import \
    make_avro_record_dataset, _maybe_shuffle_and_repeat, parse_avro
from tensorflow_io.core.python.experimental.avro_record_dataset_ops import \
    AvroRecordDataset


class BenchmarkAvroRecordDataset(benchmark_dataset.BenchmarkDataset):
    """
    Tests the perfomance of calling `make_avro_record_dataset`. This does not allow
    varying `num_parallel_reads` from `num_parallel_calls`.

    """
    def make_dataset(self, batch_size, **kwargs):
        return make_avro_record_dataset(
            file_pattern=self._filenames,
            features=self._features,
            batch_size=batch_size,
            reader_schema=self._schema_string,
            reader_buffer_size=kwargs.get('reader_buffer_size', None),
            num_epochs=self._num_epochs,
            shuffle=kwargs.get('shuffle', False),
            shuffle_buffer_size=kwargs.get("shuffle_buffer_size", None),
            num_parallel_reads=kwargs.get('num_parallel_reads', None),
            drop_final_batch=kwargs.get('drop_final_batch', False))


class BenchmarkVanillaAvroRecordDataset(benchmark_dataset.BenchmarkDataset):
    """
    Tests the performance of calling AvroRecordDataset directly.
    This allows varying `num_parallel_reads` and `num_parallel_calls`
    as well as `block_length`
    """
    def make_dataset(self, batch_size, **kwargs):
        dataset = AvroRecordDataset(
            filenames=self._filenames,
            reader_schema=self._schema_string,
            buffer_size=kwargs.get('reader_buffer_size', None),
            num_parallel_reads=kwargs.get('num_parallel_reads', None),
            num_parallel_calls=kwargs.get('num_parallel_calls', None),
            block_length=kwargs.get('block_length', None),
            deterministic=kwargs.get('deterministic', None)
        )
        dataset = _maybe_shuffle_and_repeat(dataset,
                                            self._num_epochs,
                                            kwargs.get('shuffle', False),
                                            kwargs.get("shuffle_buffer_size", None),
                                            kwargs.get("shuffle_seed", None)
                                            )
        drop_final_batch = self._num_epochs is None
        num_parallel_parser_calls = kwargs.get('num_parallel_parser_calls', None)
        dataset = dataset.batch(batch_size, drop_remainder=drop_final_batch)
        dataset = dataset.map(
            lambda data: parse_avro(
                serialized=data, reader_schema=self._schema_string, features=self._features
            ),
            num_parallel_calls=num_parallel_parser_calls,
        )
        return dataset
