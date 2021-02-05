import pytest

from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.ops import parsing_ops
from tests.test_benchmark.benchmark_dataset import BenchmarkDataWrapper
from tests.test_benchmark.avro.record.dataset.benchmark_avro_record_dataset \
    import BenchmarkVanillaAvroRecordDataset
from tests.test_benchmark.generator.data_generator import ExponentialIntegerDataGenerator


# Note, it is expensive to create the test data and thus we do that only once
@pytest.fixture(scope="module")
def test_setup():
    schema_name = "Sparse1DRowFormat.avsc"
    features = {
        "customer": parsing_ops.SparseFeature(
            index_key=["ids"],
            value_key="prices",
            dtype=tf_types.float32,
            size=[4000])
    }
    benchmark_data = BenchmarkVanillaAvroRecordDataset(schema_name=schema_name,
                                                       features=features,
                                                       num_data=4 * 1024,
                                                       num_parts=2,
                                                       num_epochs=1)

    def _constrain_generation(generator):
        generator.set_data_generator_for_all_array_num(
            ExponentialIntegerDataGenerator(beta=5, max_val=25))
        return generator

    benchmark_data.setup(constrain_generation=_constrain_generation)

    yield BenchmarkDataWrapper(benchmark_data=benchmark_data)

    benchmark_data.cleanup()


def test_dataset(test_setup):
    # Beta = 5 and max = 25
    # Total number of samples: 4k
    # Elapsed time in sec  3.6692323110764846
    # Samples per second  111630.98034526764
    # Total number of samples: 64k
    # Elapsed time in sec  68.04039667989127
    # Samples per second  96319.25032466569
    # Beta = 25 and max = 500
    # Total number of samples: 4k
    # Elapsed time in sec  14.67585639609024
    # Samples per second  27909.785224467076
    # Total number of samples: 64k
    # Elapsed time in sec  278.6533908479614
    # Samples per second  23518.823797754423
    test_setup.benchmark_data.perf_dataset(batch_size=512)
