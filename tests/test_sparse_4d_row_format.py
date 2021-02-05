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
    schema_name = "Sparse4DRowFormat.avsc"
    features = {
        "customer": parsing_ops.SparseFeature(
            index_key=["ids", "pages", "formats", "genres"],
            value_key="prices",
            dtype=tf_types.float32,
            size=[4000, 400, 400, 400])
    }
    benchmark_data = BenchmarkVanillaAvroRecordDataset(schema_name=schema_name,
                                                       features=features,
                                                       num_data=4 * 1024,
                                                       num_parts=8,
                                                       num_epochs=1)

    def _constrain_generation(generator):
        generator.set_data_generator_for_all_array_num(
            data_generator=ExponentialIntegerDataGenerator(
                beta=5, max_val=25))
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
    # The checking of the dimensions only happens in eager mode
    # To get debug information in the native code turn on flags
    # Use this to trouble shoot eager mode
    # benchmark_data = BenchmarkDataset(schema_name=schema_name,
    #                                   features=features,
    #                                   num_data=4,
    #                                   num_parts=1,
    #                                   num_epochs=1)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '50'
    #
    # beta = 5 and max = 25
    # Total number of samples: 4k
    # Elapsed time in sec  17.79681098193396
    # Samples per second  46030.7186962649
    # Total number of samples: 64k
    # Elapsed time in sec  8.164633967913687
    # Samples per second  50167.58884840312
    # beta = 25 and max = 500
    # Total number of samples: 4k
    # Elapsed time in sec  29.973692820058204
    # Samples per second  13665.316531365073
    # Total number of samples: 64k
    # Elapsed time in sec  516.6229701150442
    # Samples per second  12685.459956495182
    test_setup.benchmark_data.perf_dataset(batch_size=512)
