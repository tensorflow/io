
import tests.test_benchmark.benchmark_dataset as benchmark_dataset

from tensorflow_io.core.python.ops.avro_dataset_ops import \
    make_avro_dataset


class BenchmarkAvroDataset(benchmark_dataset.BenchmarkDataset):
    def make_dataset(self, batch_size, **kwargs):
        return make_avro_dataset(
            filenames=self._filenames,
            reader_schema=self._schema_string,
            features=self._features,
            num_parallel_calls=kwargs.get('num_parallel_calls', 4),
            batch_size=batch_size,
            shuffle=kwargs.get('shuffle', False),
            num_epochs=self._num_epochs,
            num_parallel_reads=kwargs.get('num_parallel_reads', 4))
