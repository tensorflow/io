# TensorFlow I/O JSON Datasets

[JSON (JavaScript Object Notation)](https://www.json.org/) is an open-standard
file format that uses human-readable text to transmit data objects consisting
of key–value pairs and array data types (or any other serializable value). JSON
format is widely used in machine learning and data science.

TensorFlow I/O JSON dataset supports the **record** arrays in JSON, here
is an example with 3 columns and 2 records:

```json
[
  {
    "floatfeature": 1.1,
    "integerfeature": 2,
    "floatlabel": 1.2
  },
  {
    "floatfeature": 2.1,
    "integerfeature": 3,
    "floatlabel": 1.1  
  }
]
```

## JSONDataset and tf.data

JSON Dataset is a subclass of tf.data, so it fully conforms to TensorFlow's data
pipeline (without any intermediate conversion), especially for TF 2.0.

```python
import tensorflow as tf
import tensorflow_io.json as json_io

dataset = json_io.JSONDataset(
    filename, ["floatfeature", "integerfeature"], dtype=[tf.float64, tf.int64]
).apply(tf.data.experimental.unbatch())

iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
next_element = iterator.get_next()

with tf.Session() as sess:
  for i in range(2):
    print(sess.run(next_element))
```

## Custom Ops

There are 2 different custom operators (
[Eager Execution](https://www.tensorflow.org/guide/eager) support only)
supported by JSON Datasets: `list_json_columns` and `read_json`.

### `list_json_columns`

`list_json_columns` is used to fetch all the column names of the JSON format
files. Here is an example:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
# Enable Eager Execution.
tf.enable_eager_execution()
import tensorflow_io.json as json_io

# Assume `example.json` is a JSON format file that contains the content of the
# example above.
filename = "example.json"
cols = json_io.list_json_columns(filename)
```

### `read_json`

`read_json` can be used to read a single column of the JSON file into a
Tensor by specifying column name. Here is an example:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
# Enable Eager Execution.
tf.enable_eager_execution()
import tensorflow_io.json as json_io

# Assume `example.json` is a JSON format file that contains the content of the
# example above.
filename = "example.json"
cols = json_io.list_json_columns(filename)
float_feature = json_io.read_json(
    feature_filename,
    feature_cols["floatfeature"])
```

## JSON Datasets with Keras

JSON Dataset could be used directly with tf.keras, which is the recommended high level API for TF 2.0.
Here is an example of how to use JSON Datasets to parse records from JSON file
and pass them into tf.Keras for training machine learning models.

This example uses the same dataset but in JSON format as this
[TensorFlow Custom Training Walkthrough](https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough)
does, and ties to solve the classic
[Iris classification problem](https://en.wikipedia.org/wiki/Iris_flower_data_set)
by using the `tf.Keras`.

The dataset are as following:
The `iris.json` contains 4-feature records of the flower measurements.

```json
[
  {"sepalLength": 6.9, "sepalWidth": 3.1, "petalLength": 5.1, "petalWidth": 2.3},
  {"sepalLength": 5.8, "sepalWidth": 2.7, "petalLength": 5.1, "petalWidth": 1.9},
  {"sepalLength": 6.8, "sepalWidth": 3.2, "petalLength": 5.9, "petalWidth": 2.3},
  {"sepalLength": 6.7, "sepalWidth": 3.3, "petalLength": 5.7, "petalWidth": 2.5},
  {"sepalLength": 6.7, "sepalWidth": 3.0, "petalLength": 5.2, "petalWidth": 2.3},
  ...
  {"sepalLength": 6.3, "sepalWidth": 2.5, "petalLength": 5.0, "petalWidth": 1.9},
  {"sepalLength": 6.5, "sepalWidth": 3.0, "petalLength": 5.2, "petalWidth": 2.0},
  {"sepalLength": 6.2, "sepalWidth": 3.4, "petalLength": 5.4, "petalWidth": 2.3},
  {"sepalLength": 5.9, "sepalWidth": 3.0, "petalLength": 5.1, "petalWidth": 1.8}
]
```

The `species.json` contains the label: it's an integer value of 0, 1, or 2 that
corresponds to a flower name.

```json
[
  {"species":1},
  {"species":1},
  {"species":1},
  {"species":2},
  {"species":2},
  {"species":2},
  ...
  {"species":3},
  {"species":3},
  {"species":3}
]
```

Then the `tf.data.Dataset` can be created easily:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
# Enable Eager Execution.
tf.enable_eager_execution()
import tensorflow_io.json as json_io

  ## Read JSON files.
  feature_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "iris.json")
  label_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "species.json")

  feature_cols = ["sepalLength", "sepalWidth", "petalLength", "petalWidth"]
  label_cols = ["species"]

  feature_dataset = json_io.JSONDataset(feature_filename, feature_cols)

  label_dataset = json_io.JSONDataset(label_filename, label_cols)

  dataset = tf.data.Dataset.zip((
      feature_dataset,
      label_dataset
  ))

  dataset = dataset.map(pack_features_vector)

    def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features), axis=1)
    return features, labels

```

Then, the
[same model](https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#create_a_model_using_keras)
mentioned in the
[TensorFlow Custom Training Walkthrough](https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough)
can be built as following:

```python
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
      tf.keras.layers.Dense(10, activation=tf.nn.relu),
      tf.keras.layers.Dense(3)
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  model.fit(dataset, epochs=5)
```
