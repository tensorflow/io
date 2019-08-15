# TensorFlow i/o JSON Datasets
[JSON (JavaScript Object Notation)](https://www.json.org/) is an open-standard
file format that uses human-readable text to transmit data objects consisting
of key–value pairs and array data types (or any other serializable value). JSON 
format is widely used in machine learning and data science.

TensorFlow i/o JSON support mainly focuses the **record** array in JSON, here is
an example:
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

## Custom Ops
There are 2 different custom operators ([Eager Execution]
(https://www.tensorflow.org/guide/eager) support only) supported by JSON Datasets: 
`list_json_columns` and `read_json`.

### `list_json_columns`
`list_json_columns` is used to fetch all the column names of the JSON format files.
Here is an example:
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
`read_json` can be used to read a specifed column of the JSON file into a Tensor.
Here is an example:
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
cols = json_io.read_json(filename)
float_feature = json_io.read_json(
    feature_filename,
    feature_cols["floatfeature"])
```
