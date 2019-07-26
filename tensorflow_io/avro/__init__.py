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
"""Avro Dataset.

@@AvroDataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(fraudies): See how we can leverage tf.export here for V1/V2 compatibility
from tensorflow_io.avro.python.ops.avro_dataset import make_avro_dataset_v1, \
    make_avro_dataset

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    "make_avro_dataset_v1",
    "make_avro_dataset",
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
