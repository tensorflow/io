#' Create a Dataset from LibSVM files.
#'
#' @param file_names A `tf.string` tensor containing one or more filenames.
#' @param num_features The number of features.
#' @param dtype The type of the output feature tensor. Default to `tf.float32`.
#' @param label_dtype The type of the output label tensor. Default to
#'   `tf.int64`.
#' @param batch_size An integer representing the number of records to combine in
#'   a single batch, default 1.
#' @param compression_type A `tf.string` scalar evaluating to one of `""` (no
#'   compression), `"ZLIB"`, or `"GZIP"`.
#' @param buffer_size A `tf.int64` scalar denoting the number of bytes to
#'   buffer. A value of 0 results in the default buffering values chosen based
#'   on the compression type.
#' @param num_parallel_parser_calls Number of parallel records to parse in
#'   parallel. Defaults to an automatic selection.
#' @param drop_final_batch Whether the last batch should be dropped in case its
#'   size is smaller than `batch_size`; the default behavior is not to drop the
#'   smaller batch.
#' @param prefetch_buffer_size An integer specifying the number of feature
#'   batches to prefetch for performance improvement. Defaults to auto-tune. Set
#'   to 0 to disable prefetching.
#'
#' @export
make_libsvm_dataset <- function(
  file_names,
  num_features,
  dtype = NULL,
  label_dtype = NULL,
  batch_size = 1,
  compression_type = '',
  buffer_size = NULL,
  num_parallel_parser_calls = NULL,
  drop_final_batch = FALSE,
  prefetch_buffer_size = 0) {
  dataset <- tfio_lib$libsvm$make_libsvm_dataset(
    file_names = file_names,
    num_features = num_features,
    dtype = dtype,
    label_dtype = label_dtype,
    batch_size = cast_scalar_integer(batch_size),
    compression_type = compression_type,
    buffer_size = cast_nullable_scalar_integer(buffer_size),
    num_parallel_parser_calls = cast_nullable_scalar_integer(num_parallel_parser_calls),
    drop_final_batch = cast_logical(drop_final_batch),
    prefetch_buffer_size = cast_scalar_integer(prefetch_buffer_size)
  )
  as_tf_dataset(dataset)
}
