# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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


def coord_to_int(coord, shape):
    """Convert a location in a tensor to its unique index, in row-major order.
    For example, in the 2d tensor
    [[0, 1, 2]
     [3, 4, 5]]
    The location [1, 0]  (i.e. the entry in the second row, first column) will return 3.
    """
    ret = 0
    rank = len(shape)
    for dim in range(rank):
        ret = ret * shape[dim] + coord[dim]
    return ret
