as_tf_dataset <- function (dataset, tags = NULL) {
  if (!is_dataset(dataset))
    stop("Provided dataset is not a TensorFlow Dataset")
  if (!inherits(dataset, "tf_dataset"))
    class(dataset) <- c("tf_dataset", class(dataset), tags)
  dataset
}

is_dataset <- function (x) {
  inherits(x, "tensorflow.python.data.ops.dataset_ops.Dataset") ||
  inherits(x, "tensorflow.python.data.ops.dataset_ops.DatasetV2") ||
  is_tfio_dataset(x)
}

is_tfio_dataset <- function(x) {
  grepl("tensorflow_io", class(x))
}
