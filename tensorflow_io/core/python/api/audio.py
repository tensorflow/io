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
"""tensorflow_io.audio"""

from tensorflow_io.core.python.ops.audio_ops import (  # pylint: disable=unused-import
    spectrogram,
    melscale,
    dbscale,
    remix,
    split,
    trim,
    freq_mask,
    time_mask,
    fade,
    resample,
    decode_wav,
    encode_wav,
    decode_flac,
    encode_flac,
    decode_vorbis,
    encode_vorbis,
    decode_mp3,
    encode_mp3,
    decode_aac,
    encode_aac,
    AudioIOTensor,
    AudioIODataset,
)
