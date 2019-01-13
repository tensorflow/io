#' Create a `SequenceFileDataset`.
#'
#' This function allows a user to read data from a hadoop sequence
#' file. A sequence file consists of (key value) pairs sequentially. At
#' the moment, `org.apache.hadoop.io.Text` is the only serialization type
#' being supported, and there is no compression support.
#'
#' @param filenames A `tf.string` tensor containing one or more filenames.
#'
#' @examples \dontrun{
#' dataset <- sequence_file_dataset("testdata/string.seq") %>%
#'   dataset_repeat(1)
#'
#' sess <- tf$Session()
#' iterator <- make_iterator_one_shot(dataset)
#' next_batch <- iterator_get_next(iterator)
#'
#' until_out_of_range({
#'   batch <- sess$run(next_batch)
#'   print(batch)
#' })
#' }
#'
#' @export
sequence_file_dataset <- function(filenames) {
  dataset <- tfio_lib$hadoop$SequenceFileDataset(filenames = filenames)
  as_tf_dataset(dataset)
}
