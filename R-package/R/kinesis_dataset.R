#' Creates a `KinesisDataset`.
#'
#' Kinesis is a managed service provided by AWS for data streaming.
#' This dataset reads messages from Kinesis with each message presented
#' as a `tf.string`.
#'
#' @param stream A `tf.string` tensor containing the name of the stream.
#' @param shard A `tf.string` tensor containing the id of the shard.
#' @param read_indefinitely If `True`, the Kinesis dataset will keep retry again
#'   on `EOF` after the `interval` period. If `False`, then the dataset will
#'   stop on `EOF`. The default value is `True`.
#' @param interval The interval for the Kinesis Client to wait before it tries
#'   to get records again (in millisecond).
#'
#' @export
kinesis_dataset <- function(
  stream,
  shard = "",
  read_indefinitely = TRUE,
  interval = 100000) {
  dataset <- tfio_lib$kinesis$KinesisDataset(
    stream = stream,
    shard = shard,
    read_indefinitely = cast_logical(read_indefinitely),
    interval = cast_scalar_integer(interval)
  )
  as_tf_dataset(dataset)
}
