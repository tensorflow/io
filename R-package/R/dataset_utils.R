as_tf_dataset <- function (dataset) {
  if (!is_dataset(dataset))
    stop("Provided dataset is not a TensorFlow Dataset")
  if (!inherits(dataset, "tf_dataset"))
    class(dataset) <- c("tf_dataset", class(dataset))
  dataset
}

is_dataset <- function (x) {
  inherits(x, "tensorflow.python.data.ops.dataset_ops.Dataset") || is_tfio_dataset(x)
}

is_tfio_dataset <- function(x) {
  "tensorflow_io" %in% class(x)
}
