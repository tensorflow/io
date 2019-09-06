# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Test IOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
from tensorflow.keras.models import Sequential # pylint: disable=wrong-import-position
from tensorflow.keras.layers import Dense # pylint: disable=wrong-import-position
from tensorflow.keras.layers import LSTM # pylint: disable=wrong-import-position

import tensorflow_io as tfio # pylint: disable=wrong-import-position


def test_window():
  """test_window"""
  value = [[e] for e in range(100)]
  value = tfio.IOTensor.from_tensor(tf.constant(value))
  value = value.window(3)
  expected_value = [[e, e+1, e+2] for e in range(98)]
  assert numpy.all(value.to_tensor().numpy() == expected_value)

  v = tfio.IOTensor.from_tensor(tf.constant([1, 2, 3, 4, 5]))
  v = v.window(3)
  assert numpy.all(v.to_tensor().numpy() == [[1, 2, 3], [2, 3, 4], [3, 4, 5]])

def test_window_to_dataset():
  """test_window_to_dataset"""
  value = [[e] for e in range(100)]
  value = tfio.IOTensor.from_tensor(tf.constant(value))
  value = value.window(3)
  expected_value = [[e, e+1, e+2] for e in range(98)]
  dataset = value.to_dataset()
  dataset_value = [d.numpy().tolist() for d in dataset]
  assert numpy.all(dataset_value == expected_value)

def test_io_tensor_from_tensor_with_sklearn():
  """test_io_tensor_from_tensor_with_sklearn"""

  # The test example is based on:
  # https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
  #
  # Both IOTensor and pandas/sklearn are used, to show the usage of IOTensor.
  airline_passengers_csv = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_csv", "airline-passengers.csv")

  dataframe = pandas.read_csv(
      airline_passengers_csv, usecols=[1], engine='python')

  numpy.random.seed(7)

  dataset = dataframe.values
  dataset = dataset.astype('float32')

  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)

  # split into train and test sets
  train_size = int(len(dataset) * 0.67)
  # test_size = len(dataset) - train_size
  train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
  print(len(train), len(test))

  # convert an array of values into a dataset matrix
  def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
      a = dataset[i:(i+look_back), 0]
      data_x.append(a)
      data_y.append(dataset[i + look_back, 0])
    return numpy.array(data_x), numpy.array(data_y)

  # reshape into X=t and Y=t+1
  look_back = 1
  train_x, train_y = create_dataset(train, look_back)
  test_x, test_y = create_dataset(test, look_back)

  # reshape input to be [samples, time steps, features]
  train_x = numpy.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
  test_x = numpy.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

  # create and fit the LSTM network
  model = Sequential()
  model.add(LSTM(4, input_shape=(1, look_back)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')

  #model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2)

  #########################################
  ############ IOTensor Match: ############
  train_size = int(len(dataset) * 0.67)
  #test_size = len(dataset) - train_size
  dataset = dataframe.values
  dataset = dataset.astype('float32')

  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler.fit(tfio.IOTensor.from_tensor(dataset))

  train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

  train_dataset = tfio.IOTensor.from_tensor(train)

  train_dataset = tfio.IOTensor.from_tensor(scaler.transform(train_dataset))

  train_dataset = train_dataset.window(look_back + 1)
  train_dataset = train_dataset.to_dataset()
  train_dataset = train_dataset.map(lambda e: tf.split(e, [look_back, 1]))
  train_dataset = train_dataset.map(lambda x, y: (tf.reshape(x, [1, look_back]), y))
  print("train_dataset: ", train_dataset)

  test_dataset = tfio.IOTensor.from_tensor(test)

  test_dataset = tfio.IOTensor.from_tensor(scaler.transform(test_dataset))

  test_dataset = test_dataset.window(look_back + 1)
  test_dataset = test_dataset.to_dataset()
  test_dataset = test_dataset.map(lambda e: tf.split(e, [look_back, 1]))
  test_dataset = test_dataset.map(lambda x, y: (tf.reshape(x, [1, look_back]), y))

  model.fit(train_dataset.batch(1), epochs=100, verbose=2)
  #########################################

  # make predictions
  train_predict = model.predict(train_x)
  test_predict = model.predict(test_x)

  #########################################
  ############ IOTensor Match: ############
  train_x_dataset = train_dataset.map(lambda x, y: x)
  test_x_dataset = test_dataset.map(lambda x, y: x)

  train_x_dataset = train_x_dataset.batch(1)
  test_x_dataset = test_x_dataset.batch(1)

  train_predict_dataset = model.predict(train_x_dataset)
  test_predict_dataset = model.predict(test_x_dataset)

  train_predict_dataset_value = [d.tolist() for d in train_predict_dataset]
  test_predict_dataset_value = [d.tolist() for d in test_predict_dataset]

  # Note: train_predict_dataset_value and test_predict_dataset_value
  # have one extra compared with original implementation.
  train_predict_dataset_value = train_predict_dataset_value[:-1]
  test_predict_dataset_value = test_predict_dataset_value[:-1]
  assert numpy.allclose(train_predict_dataset_value, train_predict.tolist())
  assert numpy.allclose(test_predict_dataset_value, test_predict.tolist())
  #########################################

  # invert predictions
  train_predict = scaler.inverse_transform(train_predict)
  train_y = scaler.inverse_transform([train_y])
  test_predict = scaler.inverse_transform(test_predict)
  test_y = scaler.inverse_transform([test_y])
  # calculate root mean squared error
  train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
  print('Train Score: %.2f RMSE' % (train_score))
  test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
  print('Test Score: %.2f RMSE' % (test_score))

  # shift train predictions for plotting
  train_predict_plot = numpy.empty_like(dataset)
  train_predict_plot[:, :] = numpy.nan
  train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
  # shift test predictions for plotting
  test_predict_plot = numpy.empty_like(dataset)
  test_predict_plot[:, :] = numpy.nan
  test_predict_plot[
      len(train_predict)+(look_back*2)+1:len(dataset)-1, :] = test_predict

  # plot baseline and predictions
  #plt.plot(scaler.inverse_transform(dataset))
  #plt.plot(train_predict_plot)
  #plt.plot(test_predict_plot)
  #plt.show()
