context("TensorFlow IO dataset ops")

source("utils.R")

test_succeeds("sequence_file_dataset() works successfully", {
  sequence_file_dataset("testdata/string.seq") %>%
    dataset_repeat(2)
})
