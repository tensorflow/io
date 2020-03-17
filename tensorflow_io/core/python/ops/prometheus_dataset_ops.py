# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""PrometheusDataset"""

import tensorflow as tf
from tensorflow.python.data.experimental.ops.testing import sleep
from tensorflow_io.core.python.ops import golang_ops


class PrometheusIODataset(tf.data.Dataset):
    """PrometheusIODataset"""

    def __init__(
        self, query, length, offset=None, endpoint=None, spec=None, internal=True
    ):
        """PrometheusIODataset."""
        with tf.name_scope("PrometheusIODataset"):
            assert internal

            metadata = []
            metadata.append("length=%d" % length)
            if offset is not None:
                metadata.append("offset=%d" % offset)
            if endpoint is not None:
                metadata.append("endpoint=%s" % endpoint)
            resource, metrics = golang_ops.io_prometheus_readable_init(query, metadata)
            # Construct spec in eager mode, and take spec from user in graph mode.
            if spec is None:
                spec = {}
                metrics = tf.unstack(metrics)
                jobs = list({e[0].numpy().decode() for e in metrics})
                for job in jobs:
                    entries_by_job = [e for e in metrics if e[0] == job]
                    instances = list({e[1].numpy().decode() for e in entries_by_job})
                    spec_by_job = {}
                    for instance in instances:
                        entries_by_instance = [
                            e for e in entries_by_job if e[1] == instance
                        ]
                        spec_by_instance = {}
                        for e in entries_by_instance:
                            spec_by_instance[e[2].numpy().decode()] = tf.TensorSpec(
                                [], tf.float64, e[2].numpy().decode()
                            )
                        spec_by_job[instance] = spec_by_instance
                    spec[job] = spec_by_job
            # Map spec to entries of 3 tuple (job, instance, name)
            class MetricEntry:
                def __init__(self, job, instance, name):
                    self.job = job
                    self.instance = instance
                    self.name = name

            entries = {}
            for job, spec_by_job in spec.items():
                entries_by_job = {}
                for instance, spec_by_instance in spec_by_job.items():
                    entries_by_instance = {}
                    for name in spec_by_instance.keys():
                        entries_by_instance[name] = MetricEntry(job, instance, name)
                    entries_by_job[instance] = entries_by_instance
                entries[job] = entries_by_job
            flatten = tf.nest.flatten(entries)
            metrics = [tf.stack([e.job, e.instance, e.name]) for e in flatten]
            metrics = tf.stack(metrics)

            def f(start, stop):
                timestamp, value = golang_ops.io_prometheus_readable_read(
                    resource, start=start, stop=stop, metrics=metrics
                )
                value = tf.unstack(value, num=len(flatten))
                return timestamp, tf.nest.pack_sequence_as(entries, value)

            step = 1 * 1000  # 1 second

            self._resource = resource
            start, stop = golang_ops.io_prometheus_readable_spec(resource)
            indices_start = tf.data.Dataset.range(start, stop, step)
            indices_stop = indices_start.skip(1).concatenate(
                tf.data.Dataset.from_tensor_slices([stop])
            )
            dataset = tf.data.Dataset.zip((indices_start, indices_stop))
            dataset = dataset.map(f)
            dataset = dataset.unbatch()
            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec


class PrometheusScrapeStreamIODataset(tf.data.Dataset):
    """PrometheusScrapeStreamIODataset"""

    def __init__(self, metric, endpoint, interval=None, internal=True):
        """PrometheusScrapeStreamIODataset."""
        with tf.name_scope("PrometheusScrapeStreamIODataset"):
            assert internal

            interval = 1000000 if interval is None else interval

            dataset = tf.data.experimental.Counter(start=0, step=1, dtype=tf.int64)
            dataset = dataset.map(
                lambda i: golang_ops.io_prometheus_scrape(metric, endpoint, i)
            )
            dataset = dataset.apply(sleep(interval))

            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
