import pytest

from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.ops import parsing_ops
from tests.test_benchmark.benchmark_dataset import BenchmarkDataWrapper
from tests.test_benchmark.avro.record.dataset.benchmark_avro_record_dataset import (
    BenchmarkVanillaAvroRecordDataset,
)
from tests.test_benchmark.generator.data_generator import (
    ExponentialIntegerDataGenerator,
    RepeatDataGenerator,
)


# Note, it is expensive to create the test data and thus we do that only once
@pytest.fixture(scope="module")
def test_setup():
    schema_name = "Sparse1DColumnFormat.avsc"
    features = {
        "customer": parsing_ops.SparseFeature(
            index_key=["@customer.ids[*]"],
            value_key="@customer.prices[*]",
            dtype=tf_types.float32,
            size=[4000],
        )
    }
    benchmark_data = BenchmarkVanillaAvroRecordDataset(
        schema_name=schema_name,
        features=features,
        num_data=4 * 1024,
        num_parts=2,
        num_epochs=1,
    )

    def _constrain_generation(generator):
        generator.set_data_generator_for_all_array_num(
            RepeatDataGenerator(
                data_generator=ExponentialIntegerDataGenerator(beta=5, max_val=25),
                repeat_num=2,
            )
        )
        return generator

    benchmark_data.setup(constrain_generation=_constrain_generation)
    yield BenchmarkDataWrapper(benchmark_data=benchmark_data)

    benchmark_data.cleanup()


def test_dataset(test_setup):
    # beta = 5 and max = 25
    # Total number of samples: 4k
    # Elapsed time in sec  2.6521381039638072
    # Samples per second  154441.42949713816
    # Total number of samples: 64k
    # Elapsed time in sec  44.263173635001294
    # Samples per second  148059.87600531455
    # beta = 25 and max = 500
    # Total number of samples: 4k
    # Elapsed time in sec  8.57687106297817
    # Samples per second  47756.34342552113
    # Total number of samples: 64k
    # Elapsed time in sec  149.99831869103946
    # Samples per second  43691.15638888489
    test_setup.benchmark_data.perf_dataset(batch_size=512)
