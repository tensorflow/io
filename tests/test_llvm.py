# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Test LLVM"""

import os

import tensorflow as tf
import tensorflow_io as tfio

# Placeholder for test case, output is available with TFIO_GRAPH_DEBUG=true
def test_f():
    """test_f"""

    @tf.function
    def f(v):
        v = tfio.audio.decode_wav(v, dtype=tf.int16)
        v = v + 1
        v = tfio.audio.encode_wav(v, 44100)
        return v

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    content = tf.io.read_file(path)

    _ = f(content)
