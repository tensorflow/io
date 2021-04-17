# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""
Test ORCDataset
"""

import os
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio


def test_orc_input():
    """test_pcap_input
  """
    print("Testing ORCDataset")
    orc_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_orc", "iris.orc"
    )

    dataset = tfio.IODataset.from_orc(orc_filename, capacity=15).batch(1)
    packets_total = 0
    for v in dataset:
        packets_total += 1
    assert packets_total == 150


def test_orc_keras():
    """Test case for ORCDataset with Keras"""
    orc_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_orc", "iris.orc"
    )

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    label_cols = ["species"]

    feature_dataset = tfio.IODataset.from_orc(orc_filename, columns=feature_cols)

    label_dataset = tfio.IODataset.from_orc(orc_filename, columns=label_cols)

    @tf.function
    def species_float_conversion(x):
        if x == "virginica":
            return 1.0
        if x == "versicolor":
            return 2.0
        if x == "setosa":
            return 3.0
        return 4.0

    label_dataset = label_dataset.map(species_float_conversion)
    dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))
    dataset = dataset.batch(1)

    def pack_features_vector(features, labels):
        """Pack the features into a single array."""
        features = tf.stack(list(features), axis=1)
        return features, labels

    dataset = dataset.map(pack_features_vector)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                10, activation=tf.nn.relu, input_shape=(4,)
            ),  # input shape required
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(3),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(dataset, epochs=5)


if __name__ == "__main__":
    test.main()
