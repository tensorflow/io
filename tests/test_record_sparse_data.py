from collections import namedtuple
import os
import pytest
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.ops import parsing_ops
from tests.test_benchmark.benchmark_dataset import BenchmarkDataWrapper
from tests.test_benchmark.avro.record.dataset.benchmark_avro_record_dataset \
    import BenchmarkAvroRecordDataset, BenchmarkVanillaAvroRecordDataset

"""
These are parameters for `avro-data-generator`.
    :param num_data: The number of data points. In this case, each datapoint is an Avro record
    :param num_part: The number of part files.
    :param num_epochs: The number of epochs that tells AvroRecordDataset whether to repeat.
"""
PerfParams = namedtuple('PerfParams', ['num_data', 'num_parts', 'num_epochs'])
AVRO_PARSER_NUM_MINIBATCHES = None

# Note, it is expensive to create the test data and thus we do that only once
@pytest.fixture(scope="module", params=[
    PerfParams(8*1024, 10, 10),
], ids=["small"])
def test_setup(request):
    schema_name = "SparseData.avsc"
    features = {
      "friends": parsing_ops.SparseFeature(index_key="id",
                                           value_key="salary",
                                           dtype=tf_types.float32,
                                           size=4000)
    }
    # More meaningful numbers are
    #  num_data=128*1024
    #  num_parts=16
    #  num_epochs=10
    #  Reduced numbers for tests
    benchmark_data = BenchmarkAvroRecordDataset(schema_name=schema_name,
                                                features=features,
                                                num_data=request.param.num_data,
                                                num_parts=request.param.num_parts,
                                                num_epochs=request.param.num_epochs)
    benchmark_data.setup()

    yield BenchmarkDataWrapper(benchmark_data=benchmark_data)

    benchmark_data.cleanup()


@pytest.fixture
def test_setup_vanilla():
    schema_name = "DenseData.avsc"
    features = {
        "employed": parsing_ops.FixedLenFeature([], tf_types.bool),
        "age": parsing_ops.FixedLenFeature([], tf_types.int32),
        "id": parsing_ops.FixedLenFeature([], tf_types.int64),
        "salary": parsing_ops.FixedLenFeature([], tf_types.float32),
        "altitude": parsing_ops.FixedLenFeature([], tf_types.float64),
        "name": parsing_ops.FixedLenFeature([], tf_types.string)
    }
    # More meaningful numbers are
    #  num_data=64*1024
    #  num_parts=8
    #  num_epochs=10
    #  Reduced numbers for tests
    benchmark_data = BenchmarkVanillaAvroRecordDataset(schema_name=schema_name,
                                                       features=features,
                                                       num_data=256 * 1024,
                                                       num_parts=16,
                                                       num_epochs=10)
    benchmark_data.setup()

    yield BenchmarkDataWrapper(benchmark_data=benchmark_data)

    benchmark_data.cleanup()


# These run 2 batch x 2 num_parallel_reads x 2 test_setup = 8 benchmarking tests
@pytest.mark.parametrize("batch_size,", [512])
@pytest.mark.parametrize("num_parallel_reads", [8])
def test_dataset(batch_size, num_parallel_reads, test_setup):
    # More meaningful numbers are
    #  batch_size=8192
    #  num_parallel_reads=16
    #  num_parallel_calls=16
    # Reduced numbers for tests
    test_setup.benchmark_data.perf_dataset(batch_size=batch_size,
                                           num_parallel_reads=num_parallel_reads)


@pytest.mark.benchmark(
    group="batch_8192_block_16",
)
@pytest.mark.parametrize("num_parallel_reads,num_parallel_calls,num_parallel_parser_calls,num_minibatches",
                         [(16, 8, 16, 1), (16, 8, 32, 1), (16, 8, 8, 1), (16, 8, 1, 1),
                          (16, 8, 1, 5), (16, 8, 4, 5), (16, 8, 8, 5)])
def test_vary_parallel_reads_calls(num_parallel_reads,
                                   num_parallel_calls,
                                   num_parallel_parser_calls,
                                   num_minibatches,
                                   test_setup_vanilla,
                                   benchmark):
    """
    This tests the read and parse performance.
    Read params:
    1. num_parallel_reads: number of files to read(cycle_length)
    2. num_parallel_calls: number of threads to spawn
    Parse params:
    1. num_parallel_parser_calls: number of batches to parse in parallel
    2. num_minibatches: number of minibatches to parse in parallel(within the batches)

    TODO: Separate benchmarks for read and parse

    """
    AVRO_PARSER_NUM_MINIBATCHES = num_minibatches
    # https://pytest-benchmark.readthedocs.io/en/latest/pedantic.html
    count = benchmark.pedantic(
        target=test_setup_vanilla.benchmark_data.perf_dataset,
        args=[8192],
        iterations=2,
        rounds=3,
        kwargs={
            "num_parallel_reads": num_parallel_reads,
            "num_parallel_calls": num_parallel_calls,
            "num_parallel_parser_calls": num_parallel_parser_calls,
            "block_length": 16,
            "deterministic": False
        }
    )
    assert count > 0, f"Count: {count} must be greater than 0"


@pytest.mark.benchmark(
    group="reads_16_calls_8_pcalls_16_mbatch_1",
)
@pytest.mark.parametrize("batch_size,block_length",
                         [(8192, 16), (8192, 64), (8192 * 4, 16), (8192 * 4, 16)])
def test_vary_batch_block(batch_size, block_length, test_setup_vanilla, benchmark):
    AVRO_PARSER_NUM_MINIBATCHES = 5
    # https://pytest-benchmark.readthedocs.io/en/latest/pedantic.html
    count = benchmark.pedantic(
        target=test_setup_vanilla.benchmark_data.perf_dataset,
        args=[batch_size],
        iterations=2,
        rounds=1,
        kwargs={
            "num_parallel_reads": 16,
            "num_parallel_calls": 8,
            "num_parallel_parser_calls": 16,
            "block_length": block_length,
            "deterministic": False
        }
    )
    assert count > 0, f"Count: {count} must be greater than 0"

