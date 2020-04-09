#' @title ArrowDataset
#'
#' @description An Arrow Dataset from record batches in memory, or a Pandas DataFrame.
#'
#' @details 
#'
#' @param serialized_batches serialized_batches
#' @param columns columns
#' @param output_types output_types
#' @param output_shapes output_shapes
#' @param batch_size batch_size
#' @param batch_mode batch_mode
#' @param arrow_buffer arrow_buffer
#'
#' @export
ArrowDataset <- function(serialized_batches, columns, output_types, output_shapes = NULL, batch_size = NULL, batch_mode = "keep_remainder", arrow_buffer = NULL) {

  python_function_result <- tf_io$arrow$ArrowDataset(
    serialized_batches = serialized_batches,
    columns = columns,
    output_types = output_types,
    output_shapes = output_shapes,
    batch_size = batch_size,
    batch_mode = batch_mode,
    arrow_buffer = arrow_buffer
  )

}
#' @title ArrowFeatherDataset
#'
#' @description An Arrow Dataset for reading record batches from Arrow feather files.
#'
#' @details Feather is a light-weight columnar format ideal for simple writing of
#' Pandas DataFrames. Pyarrow can be used for reading/writing Feather files,
#' see https://arrow.apache.org/docs/python/ipc.html#feather-format
#'
#' @param filenames filenames
#' @param columns columns
#' @param output_types output_types
#' @param output_shapes output_shapes
#' @param batch_size batch_size
#' @param batch_mode batch_mode
#'
#' @export
ArrowFeatherDataset <- function(filenames, columns, output_types, output_shapes = NULL, batch_size = NULL, batch_mode = "keep_remainder") {

  python_function_result <- tf_io$arrow$ArrowFeatherDataset(
    filenames = filenames,
    columns = columns,
    output_types = output_types,
    output_shapes = output_shapes,
    batch_size = batch_size,
    batch_mode = batch_mode
  )

}
#' @title ArrowStreamDataset
#'
#' @description An Arrow Dataset for reading record batches from an input stream.
#'
#' @details Currently supported input streams are a socket client or stdin.
#'
#' @param endpoints endpoints
#' @param columns columns
#' @param output_types output_types
#' @param output_shapes output_shapes
#' @param batch_size batch_size
#' @param batch_mode batch_mode
#'
#' @export
ArrowStreamDataset <- function(endpoints, columns, output_types, output_shapes = NULL, batch_size = NULL, batch_mode = "keep_remainder") {

  python_function_result <- tf_io$arrow$ArrowStreamDataset(
    endpoints = endpoints,
    columns = columns,
    output_types = output_types,
    output_shapes = output_shapes,
    batch_size = batch_size,
    batch_mode = batch_mode
  )

}
#' @title list_feather_columns
#'
#' @description list_feather_columns
#'
#' @details 
#'
#' @param filename filename
#'
#' @export
list_feather_columns <- function(filename) {

  python_function_result <- tf_io$arrow$list_feather_columns(
    filename = filename
  )

}
