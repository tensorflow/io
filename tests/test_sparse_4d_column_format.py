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
    schema_name = "Sparse4DColumnFormat.avsc"
    features = {
        "customer": parsing_ops.SparseFeature(
            index_key=[
                "@customer.ids[*]",
                "@customer.pages[*]",
                "@customer.formats[*]",
                "@customer.genres[*]",
            ],
            value_key="@customer.prices[*]",
            dtype=tf_types.float32,
            size=[4000, 400, 400, 400],
        )
    }
    benchmark_data = BenchmarkVanillaAvroRecordDataset(
        schema_name=schema_name,
        features=features,
        num_data=4 * 1024,
        num_parts=8,
        num_epochs=1,
    )

    def _constrain_generation(generator):
        generator.set_data_generator_for_all_array_num(
            RepeatDataGenerator(
                data_generator=ExponentialIntegerDataGenerator(beta=5, max_val=50),
                repeat_num=5,
            )
        )
        return generator

    benchmark_data.setup(constrain_generation=_constrain_generation)

    yield BenchmarkDataWrapper(benchmark_data=benchmark_data)

    benchmark_data.cleanup()


def test_dataset(test_setup):
    # TODO(fraudies): The merge_sparse op is not used properly or defined
    #  properly for n>1 indices
    # Each index introduces an entry in the dense shape, here creating a
    # dense shape of [? ? ? ? 4000 400 400 400] with rank 8, the proper
    # rank is 5
    # For debugging turn on eager mode
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '50'
    #
    # beta = 5 and max = 25
    # Total number of samples: 4k
    # Elapsed time in sec  13.796838010079227
    # Samples per second  59375.92362840939
    # Total number of samples: 64k
    # Elapsed time in sec  102.91108757595066
    # Samples per second  63682.1566496739
    # beta = 25 and max = 500
    # Total number of samples: 4k
    # Elapsed time in sec  18.916356634930708
    # Samples per second  21653.218318142604
    # Total number of samples: 64k
    # Elapsed time in sec  333.053244507988
    # Samples per second  19677.334204269606
    test_setup.benchmark_data.perf_dataset(batch_size=512)
