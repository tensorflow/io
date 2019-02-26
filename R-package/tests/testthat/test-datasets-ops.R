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
      cache_name = "SQL_PUBLIC_TEST_CACHE", port = 10800)
  iterate_all_batches(dataset)
})

test_succeeds("parquet_dataset() works successfully", {
  skip_on_travis()
  dtypes <- tf$python$framework$dtypes
  output_types <- reticulate::tuple(
    dtypes$bool, dtypes$int32, dtypes$int64, dtypes$float32, dtypes$float64)
  dataset <- parquet_dataset(
      filenames = list("testdata/parquet_cpp_example.parquet"),
      columns = list(0, 1, 2, 4, 5),
      output_types = output_types) %>%
    dataset_repeat(2)
  iterate_all_batches(dataset)
})

test_succeeds("webp_dataset() works successfully", {
  skip_on_travis()
  dataset <- webp_dataset(
      filenames = list("testdata/sample.webp")) %>%
    dataset_repeat(2)
  iterate_all_batches(dataset)
})

test_succeeds("video_dataset() works successfully", {
  skip_on_travis()
  dataset <- video_dataset(
    filenames = list("testdata/small.mp4")) %>%
    dataset_repeat(2)
  iterate_all_batches(dataset)
})

test_succeeds("lmdb_dataset() works successfully", {
  skip_on_travis()
  dataset <- lmdb_dataset(
    filenames = list("testdata/data.mdb")) %>%
    dataset_repeat(2)
  iterate_all_batches(dataset)
})

test_succeeds("tiff_dataset() works successfully", {
  skip_on_travis()
  dataset <- tiff_dataset(
      filenames = list("testdata/small.tiff")) %>%
    dataset_repeat(2)
  iterate_all_batches(dataset)
})
