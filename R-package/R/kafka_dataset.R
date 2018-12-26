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
    eof = eof,
    timeout = as.integer(timeout)
  )
  as_tf_dataset(dataset)
}
