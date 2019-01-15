#' Create a `VideoDataset` that reads the video file.
#'
#' This allows a user to read data from a video file with ffmpeg. The output of
#' VideoDataset is a sequence of (height, weight, 3) tensor in rgb24 format.
#'
#' @param filenames A `tf.string` tensor containing one or more filenames.
#'
#' @examples \dontrun{
#' dataset <- video_dataset(
#'     filenames = list("testdata/small.mp4")) %>%
#'   dataset_repeat(2)
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
video_dataset <- function(filenames) {
  dataset <- tfio_lib$video$VideoDataset(filenames = filenames)
  as_tf_dataset(dataset)
}
