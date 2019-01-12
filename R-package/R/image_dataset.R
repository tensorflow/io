#' Create a `WebPDataset`.
#'
#' A WebP Image File Dataset that reads the WebP file.
#'
#' @param filenames A `tf.string` tensor containing one or more filenames.
#'
#' @export
webp_dataset <- function(filenames) {
  dataset <- tfio_lib$image$WebPDataset(filenames = filenames)
  as_tf_dataset(dataset)
}
