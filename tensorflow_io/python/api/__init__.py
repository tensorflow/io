# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""tensorflow_io"""
import os

# tensorflow_io.core.python.ops is implicitly imported (along with file system)
from tensorflow_io.python.ops.io_dataset import IODataset
from tensorflow_io.python.ops.io_tensor import IOTensor

from tensorflow_io.python.api import genome
from tensorflow_io.python.api import image
from tensorflow_io.python.api import audio
from tensorflow_io.python.api import version
from tensorflow_io.python.api import experimental
from tensorflow_io.python.api import bigtable

if os.environ.get("GENERATING_TF_DOCS", ""):
    # Mark these as public api for /tools/docs/build_docs.py
    from tensorflow_io import arrow
    from tensorflow_io import bigquery

del os
