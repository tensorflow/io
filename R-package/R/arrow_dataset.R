#' Creates a `ArrowFeatherDataset`.
#'
#' An Arrow Dataset for reading record batches from Arrow feather files. Feather
#' is a light-weight columnar format ideal for simple writing of Pandas
#' DataFrames.
#'
#' @param filenames A `tf.string` tensor, list or scalar containing files in
#'   Arrow Feather format.
#' @param columns A list of column indices to be used in the Dataset.
#' @param output_types Tensor dtypes of the output tensors.
#' @param output_shapes TensorShapes of the output tensors or `NULL` to infer
#'   partial.
#'
#' @examples \dontrun{
#' dataset <- arrow_feather_dataset(
#'     list('/path/to/a.feather', '/path/to/b.feather'),
#'     columns = reticulate::tuple(0L, 1L),
#'     output_types = reticulate::tuple(tf$int32, tf$float32),
#'     output_shapes = reticulate::tuple(list(), list())) %>%
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
arrow_feather_dataset <- function(
  filenames,
  columns,
  output_types,
  output_shapes = NULL) {
  dataset <- tfio_lib$arrow$ArrowFeatherDataset(
    filenames = filenames,
    # TODO: More user-friendly automatic type casting here
    # to avoid the need of `reticulate::tuple(0L, 1L)`
    columns = columns,
    output_types = output_types,
    output_shapes = output_shapes
  )
  as_tf_dataset(dataset)
}

#' Creates a `ArrowStreamDataset`.
#'
#' An Arrow Dataset for reading record batches from an input stream. Currently
#' supported input streams are a socket client or stdin.
#'
#' @param host A `tf.string` tensor or string defining the input stream.
#'   For a socket client, use "<HOST_IP>:<PORT>", for stdin use "STDIN".
#' @param columns A list of column indices to be used in the Dataset.
#' @param output_types Tensor dtypes of the output tensors.
#' @param output_shapes TensorShapes of the output tensors or `NULL` to infer
#'   partial.
#'
#' @examples \dontrun{
#' dataset <- arrow_stream_dataset(
#'     host,
#'     columns = reticulate::tuple(0L, 1L),
#'     output_types = reticulate::tuple(tf$int32, tf$float32),
#'     output_shapes = reticulate::tuple(list(), list())) %>%
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
arrow_stream_dataset <- function(
  host,
  columns,
  output_types,
  output_shapes = NULL) {
  dataset <- tfio_lib$arrow$ArrowFeatherDataset(
    host = host,
    # TODO: More user-friendly automatic type casting here
    # to avoid the need of `reticulate::tuple(0L, 1L)`
    columns = columns,
    output_types = output_types,
    output_shapes = output_shapes
  )
  as_tf_dataset(dataset)
}

#' Create an Arrow Dataset from the given Arrow schema.
#'
#' Infer output types and shapes from the given Arrow schema and create an Arrow
#' Dataset.
#'
#' @param object An \R object.
#' @param ... Optional arguments passed on to implementing methods.
#'
#' @export
from_schema <- function(object, ...) {
  UseMethod("from_schema")
}

#' Create an Arrow Dataset for reading record batches from Arrow feather files,
#' inferring output types and shapes from the given Arrow schema.
#'
#' @param filenames A `tf.string` tensor, list or scalar containing files in
#'   Arrow Feather format.
#' @param schema Arrow schema defining the record batch data in the stream.
#' @param columns A list of column indices to be used in the Dataset.
#' @export
from_schema.tensorflow_io.arrow.python.ops.arrow_dataset_ops.ArrowFeatherDataset <- function(
  object, filenames, schema, columns = NULL) {
  object$from_schema(
    filenames = filenames,
    schema = schema,
    columns = columns
  )
}

#' Create an Arrow Dataset from an input stream, inferring output types and
#' shapes from the given Arrow schema.
#'
#' @param host A `tf.string` tensor or string defining the input stream.
#'   For a socket client, use "<HOST_IP>:<PORT>", for stdin use "STDIN".
#' @param schema Arrow schema defining the record batch data in the stream.
#' @param columns A list of column indices to be used in the Dataset.
#'
#' @export
from_schema.tensorflow_io.arrow.python.ops.arrow_dataset_ops.ArrowStreamDataset <- function(
  object, host, schema, columns = NULL) {
  object$from_schema(
    host = host,
    schema = schema,
    columns = columns
  )
}
