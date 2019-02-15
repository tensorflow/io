#' Create a `LMDBDataset`.
#'
#' This function allows a user to read data from a LMDB
#' file. A lmdb file consists of (key value) pairs sequentially.
#'
#' @param filenames A `tf.string` tensor containing one or more filenames.
#'
#' @examples \dontrun{
#' dataset <- sequence_file_dataset("testdata/data.mdb") %>%
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
lmdb_dataset <- function(filenames) {
  dataset <- tfio_lib$lmdb$LMDBDataset(filenames = filenames)
  as_tf_dataset(dataset)
}
