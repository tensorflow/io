from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple
import logging
import glob
import os
import shutil
import tempfile

from pkg_resources import resource_filename
from time import perf_counter

from tests.test_benchmark.generator.generator import Generator


class BenchmarkDataWrapper(namedtuple('BenchmarkDataWrapper',
                                      ['benchmark_data'])):
    """Capture the context information for each unit test"""


def _default_constrain_generation(generator):
    return generator


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class BenchmarkDataset:

    def __init__(self, schema_name, features, num_data=8*1024, num_parts=10,
                 num_epochs=10):
        self._schema_name = schema_name
        self._features = features
        self._num_data = num_data
        self._num_parts = num_parts
        self._num_epochs = num_epochs
        self._filenames = []
        self._schema_path = ""
        self._schema_string = ""
        self._tmp_dir = ""
        self.count = 0

    def setup(self, constrain_generation=_default_constrain_generation):
        # Load schema
        self._schema_path = resource_filename(
            "tests.test_benchmark.resources.schemas", self._schema_name)
        self._schema_string = open(self._schema_path, "rb").read()

        # Create temporary file
        self._tmp_dir = tempfile.mkdtemp()

        # Generate data
        generator = constrain_generation(Generator.create(self._schema_path))
        generator.write(output_path=self._tmp_dir, n_data=self._num_data,
                        n_part=self._num_parts)
        self._filenames = glob.glob(os.path.join(self._tmp_dir, "*.avro"))

    def cleanup(self):
        shutil.rmtree(self._tmp_dir)

    def perf_dataset(self, batch_size, **kwargs):
        dataset = self.make_dataset(batch_size, **kwargs)
        start_time = perf_counter()
        num_samples = 0
        for _ in iter(dataset):
            num_samples += batch_size
        duration = perf_counter() - start_time
        log.debug("Avro dataset")
        log.debug("Elapsed time in sec ", duration)
        log.debug("Samples per second ", num_samples/duration)
        self.count += 1
        return self.count

    @abc.abstractmethod
    def make_dataset(self, batch_size, **kwargs):
        pass
