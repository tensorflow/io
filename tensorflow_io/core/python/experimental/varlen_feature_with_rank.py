# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""VarLenFeatureWithRank"""

import tensorflow as tf


class VarLenFeatureWithRank:
    """
    A class used to represent VarLenFeature with rank.
    This allows rank to be passed by users, and when parsing,
    rank will be used to determine the shape of sparse feature.
    User should use this class as opposed to VarLenFeature
    when defining features of data.
    """

    def __init__(self, dtype: tf.dtypes.DType, rank: int = 1):
        self.__dtype = dtype
        self.__rank = rank

    @property
    def rank(self):
        return self.__rank

    @property
    def dtype(self):
        return self.__dtype
