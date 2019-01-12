context("TensorFlow IO dataset ops")

source("utils.R")

test_succeeds("sequence_file_dataset() works successfully", {
  sequence_file_dataset("testdata/string.seq") %>%
    dataset_repeat(2)
})

test_succeeds("kafka_dataset() works successfully", {
  dtypes <- tf$python$framework$dtypes
  array_ops <- tf$python$array_ops

  topics <- array_ops$placeholder(dtypes$string, shape = list(NULL))
  num_epochs <- array_ops.placeholder(dtypes$int64, shape = list())
  batch_size <- array_ops.placeholder(dtypes$int64, shape = list())

  repeat_dataset = kafka_dataset(
      topics = topics, group = "test", eof = TRUE) %>%
    dataset_repeat(num_epochs)
})

test_succeeds("ignite_dataset() works successfully", {
  repeat_dataset = ignite_dataset(
      cache_name = "SQL_PUBLIC_TEST_CACHE", port = 42300)
})
