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
#' @export
parquet_dataset <- function(filenames, columns, output_types) {
  dataset <- tfio_lib$parquet$ParquetDataset(
    filenames = filenames,
    columns = cast_integer_list(columns),
    output_types = output_types
  )
  as_tf_dataset(dataset)
}
