#' Creates a `PubSubDataset`.
#'
#' This creates a dataset for consuming PubSub messages.
#'
#' @param subscriptions A `tf.string` tensor containing one or more
#'   subscriptions.
#' @param server The pubsub server.
#' @param eof If True, the pubsub reader will stop on EOF.
#' @param timeout The timeout value for the PubSub to wait (in millisecond).
#'   
#' @export
pubsub_dataset <- function(
  subscriptions,
  server = NULL,
  eof = FALSE,
  timeout = 1000) {
  dataset <- tfio_lib$pubsub$PubSubDataset(
    subscriptions = subscriptions,
    server = server,
    eof = cast_logical(eof),
    timeout = cast_scalar_integer(timeout)
  )
  as_tf_dataset(dataset)
}
