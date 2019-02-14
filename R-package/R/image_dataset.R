#' Create a `WebPDataset`.
#'
#' A WebP Image File Dataset that reads the WebP file.
#'
#' @param filenames A `tf.string` tensor containing one or more filenames.
#'
#' @examples \dontrun{
#' dataset <- webp_dataset(
#'     filenames = list("testdata/sample.webp")) %>%
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
webp_dataset <- function(filenames) {
  dataset <- tfio_lib$image$WebPDataset(filenames = filenames)
  as_tf_dataset(dataset)
}
#' Create a `TIFFDataset`.
#'
#' A TIFF Image File Dataset that reads the TIFF file.
#'
#' @param filenames A `tf.string` tensor containing one or more filenames.
#'
#' @examples \dontrun{
#' dataset <- tiff_dataset(
#'     filenames = list("testdata/small.tiff")) %>%
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
tiff_dataset <- function(filenames) {
  dataset <- tfio_lib$image$TIFFDataset(filenames = filenames)
  as_tf_dataset(dataset)
}
