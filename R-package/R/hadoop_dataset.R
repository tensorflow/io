#' Create a `SequenceFileDataset`.
#'
#' This function allows a user to read data from a hadoop sequence
#' file. A sequence file consists of (key value) pairs sequentially. At
#' the moment, `org.apache.hadoop.io.Text` is the only serialization type
#' being supported, and there is no compression support.
#'
#' @param filenames A `tf.string` tensor containing one or more filenames.
#'
#' @export
sequence_file_dataset <- function(filenames) {
  dataset <- tfio_lib$hadoop$SequenceFileDataset(filenames = filenames)
  as_tf_dataset(dataset)
}
