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
"""Tests for CSV"""


import os
import tempfile

import numpy as np

import pandas as pd

import tensorflow_io as tfio  # pylint: disable=wrong-import-position


def test_csv_format():
    """test_csv_format"""
    data = {
        "bool": np.asarray([e % 2 for e in range(100)], np.bool),
        "int64": np.asarray(range(100), np.int64),
        "double": np.asarray(range(100), np.float64),
    }
    df = pd.DataFrame(data).sort_index(axis=1)
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        df.to_csv(f, index=False)

    df = pd.read_csv(f.name)

    csv = tfio.IOTensor.from_csv(f.name)
    for column in df.columns:
        assert csv(column).shape == [100]
        assert csv(column).dtype == column
        assert np.all(csv(column).to_tensor().numpy() == data[column])

    os.unlink(f.name)


def test_null_csv_format():
    """test_null_csv_format"""
    # cat tests/test_csv/null.csv
    # C1,C2,C3
    # 1,2,3
    # 4,NaN,6
    # 7,8,9
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_csv", "null.csv"
    )
    csv = tfio.IOTensor.from_csv(csv_path)
    assert np.all(csv.isnull("C2").to_tensor().numpy() == [False, True, False])


def test_str_csv_format():
    """test_str_csv_format"""
    # CSV from https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
    # First 10 entries:
    # "Month","Passengers"
    # "1949-01",112
    # "1949-02",118
    # "1949-03",132
    # "1949-04",129
    # "1949-05",121
    # "1949-06",135
    # "1949-07",148
    # "1949-08",148
    # "1949-09",136
    # "1949-10",119
    # See https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_csv", "airline-passengers.csv"
    )
    csv = tfio.IOTensor.from_csv(csv_path)
    assert np.all(
        csv("Month").to_tensor()[0:10].numpy()
        == [
            b"1949-01",
            b"1949-02",
            b"1949-03",
            b"1949-04",
            b"1949-05",
            b"1949-06",
            b"1949-07",
            b"1949-08",
            b"1949-09",
            b"1949-10",
        ]
    )
    assert np.all(
        csv("Passengers").to_tensor()[0:10].numpy()
        == [
            112,
            118,
            132,
            129,
            121,
            135,
            148,
            148,
            136,
            119,
        ]
    )


if __name__ == "__main__":
    test.main()
