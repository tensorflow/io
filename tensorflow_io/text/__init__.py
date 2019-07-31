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
"""TextInput/TextOutput

@@TextOutputSequence
@@TextDataset
@@save_text
@@save_csv
@@from_csv
@@re2_full_match
@@read_text
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_io.text.python.ops.text_ops import TextOutputSequence
from tensorflow_io.text.python.ops.text_ops import TextDataset
from tensorflow_io.text.python.ops.text_ops import save_text
from tensorflow_io.text.python.ops.text_ops import save_csv
from tensorflow_io.text.python.ops.text_ops import from_csv
from tensorflow_io.text.python.ops.text_ops import re2_full_match
from tensorflow_io.text.python.ops.text_ops import read_text

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    "TextOutputSequence",
    "TextDataset",
    "save_text",
    "save_csv",
    "from_csv",
    "re2_full_match",
    "read_text",
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
