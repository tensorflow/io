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
"""Image Dataset.

@@WebPDataset
@@TIFFDataset
@@GIFDataset
@@decode_webp
@@draw_bounding_boxes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_io.image.python.ops.image_dataset_ops import WebPDataset
from tensorflow_io.image.python.ops.image_dataset_ops import TIFFDataset
from tensorflow_io.image.python.ops.image_dataset_ops import GIFDataset
from tensorflow_io.image.python.ops.image_dataset_ops import decode_webp
from tensorflow_io.image.python.ops.image_dataset_ops import draw_bounding_boxes

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    "WebPDataset",
    "TIFFDataset",
    "GIFDataset",
    "decode_webp",
    "draw_bounding_boxes",
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
