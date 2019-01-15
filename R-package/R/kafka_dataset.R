#' Creates a `KafkaDataset`.
#'
#' @param topics A `tf.string` tensor containing one or more subscriptions, in
#'   the format of `[topic:partition:offset:length]`, by default length is -1
#'   for unlimited.
#' @param servers A list of bootstrap servers.
#' @param group The consumer group id.
#' @param eof If True, the kafka reader will stop on EOF.
#' @param timeout The timeout value for the Kafka Consumer to wait (in
#'   millisecond).
#'
#' @examples \dontrun{
#' dataset <- kafka_dataset(
#'     topics = list("test:0:0:4"), group = "test", eof = TRUE) %>%
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
kafka_dataset <- function(
  topics,
  servers = "localhost",
  group = "",
  eof = FALSE,
  timeout = 1000) {
  dataset <- tfio_lib$kafka$KafkaDataset(
    topics = topics,
    servers = servers,
    group = group,
    eof = cast_logical(eof),
    timeout = cast_scalar_integer(timeout)
  )
  as_tf_dataset(dataset)
}
