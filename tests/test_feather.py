# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Feather"""


import os
import tempfile
import pytest

import tensorflow as tf
import tensorflow_io as tfio


@pytest.mark.parametrize(
    ("version"),
    [
        1,
        2,
    ],
    ids=[
        "v1",
        "v2",
    ],
)
def test_feather_format(version):
    """test_feather_format"""
    import numpy as np
    import pandas as pd

    from pyarrow import feather as pa_feather

    data = {
        "bool": np.asarray([e % 2 for e in range(100)], bool),
        "int8": np.asarray(range(100), np.int8),
        "int16": np.asarray(range(100), np.int16),
        "int32": np.asarray(range(100), np.int32),
        "int64": np.asarray(range(100), np.int64),
        "float": np.asarray(range(100), np.float32),
        "double": np.asarray(range(100), np.float64),
    }
    df = pd.DataFrame(data).sort_index(axis=1)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pa_feather.write_feather(df, f, version=version)

    feather = tfio.IOTensor.from_feather(f.name)
    for column in df.columns:
        assert feather(column).shape == [100]
        assert feather(column).dtype == column
        assert np.all(feather(column).to_tensor().numpy() == data[column])

    os.unlink(f.name)


def test_binary_feather_format():
    """test_binary_feather_format"""
    import numpy as np
    import pandas as pd

    from pyarrow import feather as pa_feather
    import pyarrow as pa

    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "sample.png"
    )
    with open(local_path, "rb") as f:
        data = [f.read()]
        table = pa.Table.from_arrays([data], ["data"])

    chunk_size = 1000
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pa_feather.write_feather(table, f, chunksize=chunk_size)

    feather = tfio.IOTensor.from_feather(f.name)
    assert feather("data").shape == [1]
    assert feather("data").dtype == tf.string
    assert np.all(feather("data").to_tensor().numpy() == data[0])

    os.unlink(f.name)


if __name__ == "__main__":
    test.main()
