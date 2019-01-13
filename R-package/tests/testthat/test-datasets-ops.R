context("TensorFlow IO dataset ops")

source("utils.R")

test_succeeds("sequence_file_dataset() works successfully", {
  dataset <- sequence_file_dataset("testdata/string.seq") %>%
    dataset_repeat(2)
  iterate_all_batches(dataset)
})

test_succeeds("kafka_dataset() works successfully", {
  dataset <- kafka_dataset(
      topics = list("test:0:0:4"), group = "test", eof = TRUE) %>%
    dataset_repeat(1)
  iterate_all_batches(dataset)
})

test_succeeds("ignite_dataset() works successfully", {
  skip_on_travis()
  dataset <- ignite_dataset(
      cache_name = "SQL_PUBLIC_TEST_CACHE", port = 42300)
  iterate_all_batches(dataset)
})

test_succeeds("parquet_dataset() works successfully", {
  skip_on_travis()
  dtypes <- tf$python$framework$dtypes
  constant_op <- tf$python$constant_op
  filenames <- constant_op$constant(
    list("testdata/parquet_cpp_example.parquet"), dtypes$string)
  columns <- list(0, 1, 2, 4, 5)
  output_types <- reticulate::tuple(
    dtypes$bool, dtypes$int32, dtypes$int64, dtypes$float32, dtypes$float64)
  dataset <- parquet_dataset(filenames, columns, output_types) %>%
    dataset_repeat(2)
  iterate_all_batches(dataset)
})

test_succeeds("webp_dataset() works successfully", {
  skip_on_travis()
  dtypes <- tf$python$framework$dtypes
  constant_op <- tf$python$constant_op
  filenames <- constant_op$constant(
    list("testdata/sample.webp"), dtypes$string)
  dataset <- webp_dataset(filenames) %>% dataset_repeat(2)
  iterate_all_batches(dataset)
})
