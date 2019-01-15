#' Create a `ParquetDataset`.
#'
#' This allows a user to read data from a parquet file.
#'
#' @param filenames A 0-D or 1-D `tf.string` tensor containing one or more
#'   filenames.
#' @param columns A 0-D or 1-D `tf.int32` tensor containing the columns to
#'   extract.
#' @param output_types A tuple of `tf.DType` objects representing the types of
#'   the columns returned.
#'
#' @examples \dontrun{
#' dtypes <- tf$python$framework$dtypes
#' output_types <- reticulate::tuple(
#'   dtypes$bool, dtypes$int32, dtypes$int64, dtypes$float32, dtypes$float64)
#' dataset <- parquet_dataset(
#'     filenames = list("testdata/parquet_cpp_example.parquet"),
#'     columns = list(0, 1, 2, 4, 5),
#'     output_types = output_types) %>%
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
parquet_dataset <- function(filenames, columns, output_types) {
  dataset <- tfio_lib$parquet$ParquetDataset(
    filenames = filenames,
    columns = cast_integer_list(columns),
    output_types = output_types
  )
  as_tf_dataset(dataset)
}
