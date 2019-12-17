# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Arrow Dataset.

@@ArrowDataset
@@ArrowFeatherDataset
@@ArrowStreamDataset
@@list_feather_columns
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_io.arrow.python.ops.arrow_dataset_ops import ArrowDataset
from tensorflow_io.arrow.python.ops.arrow_dataset_ops import ArrowFeatherDataset
from tensorflow_io.arrow.python.ops.arrow_dataset_ops import ArrowStreamDataset
from tensorflow_io.arrow.python.ops.arrow_dataset_ops import list_feather_columns

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    "ArrowDataset",
    "ArrowFeatherDataset",
    "ArrowStreamDataset",
    "list_feather_columns",
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
