import pytest

from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.ops import parsing_ops
from tests.test_benchmark.benchmark_dataset import BenchmarkDataWrapper
from tests.test_benchmark.avro.record.dataset.benchmark_avro_record_dataset \
    import BenchmarkVanillaAvroRecordDataset


# Note, it is expensive to create the test data and thus we do that only once
@pytest.fixture(scope="module")
def test_setup():
    schema_name = "DenseData.avsc"
    features = {
      "employed": parsing_ops.FixedLenFeature([], tf_types.bool),
      "age": parsing_ops.FixedLenFeature([], tf_types.int32),
      "id": parsing_ops.FixedLenFeature([], tf_types.int64),
      "salary": parsing_ops.FixedLenFeature([], tf_types.float32),
      "altitude": parsing_ops.FixedLenFeature([], tf_types.float64),
      "name": parsing_ops.FixedLenFeature([], tf_types.string)
    }
    benchmark_data = BenchmarkVanillaAvroRecordDataset(schema_name=schema_name,
                                                       features=features,
                                                       num_data=4 * 1024,
                                                       num_parts=8,
                                                       num_epochs=1)
    benchmark_data.setup()

    yield BenchmarkDataWrapper(benchmark_data=benchmark_data)

    benchmark_data.cleanup()


def test_dataset(test_setup):
    test_setup.benchmark_data.perf_dataset(batch_size=512)
