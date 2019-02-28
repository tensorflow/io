#' Creates a `MNISTImageDataset`.
#'
#' This creates a dataset for MNIST images.
#'
#' @param filenames A `tf.string` tensor containing one or more filenames.
#' @param compression_type A `tf.string` scalar evaluating to one
#'   of `""` (no compression), `"ZLIB"`, or `"GZIP"`.
#'
#' @export
mnist_image_dataset <- function(
  filenames,
  compression_type = NULL) {
  dataset <- tfio_lib$mnist$MNISTImageDataset(
    filenames = filenames,
    compression_type = compression_type
  )
  as_tf_dataset(dataset)
}

#' Creates a `MNISTLabelDataset`.
#'
#' This creates a dataset for MNIST labels.
#'
#' @param filenames A `tf.string` tensor containing one or more filenames.
#' @param compression_type A `tf.string` scalar evaluating to one
#'   of `""` (no compression), `"ZLIB"`, or `"GZIP"`.
#'
#' @export
mnist_label_dataset <- function(
  filenames,
  compression_type = NULL) {
  dataset <- tfio_lib$mnist$MNISTLabelDataset(
    filenames = filenames,
    compression_type = compression_type
  )
  as_tf_dataset(dataset)
}
