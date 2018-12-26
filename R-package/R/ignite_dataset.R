#' Create a `IgniteDataset`.
#'
#' Apache Ignite is a memory-centric distributed database, caching, and
#' processing platform for transactional, analytical, and streaming workloads,
#' delivering in-memory speeds at petabyte scale. This contrib package
#' contains an integration between Apache Ignite and TensorFlow. The
#' integration is based on tf.data from TensorFlow side and Binary Client
#' Protocol from Apache Ignite side. It allows to use Apache Ignite as a
#' datasource for neural network training, inference and all other
#' computations supported by TensorFlow. Ignite Dataset is based on Apache
#' Ignite Binary Client Protocol.
#'
#' @param cache_name Cache name to be used as datasource.
#' @param host Apache Ignite Thin Client host to be connected.
#' @param port Apache Ignite Thin Client port to be connected.
#' @param local Local flag that defines to query only local data.
#' @param part Number of partitions to be queried.
#' @param page_size Apache Ignite Thin Client page size.
#' @param username Apache Ignite Thin Client authentication username.
#' @param password Apache Ignite Thin Client authentication password.
#' @param certfile File in PEM format containing the certificate as well as any
#'   number of CA certificates needed to establish the certificate's
#'   authenticity.
#' @param keyfile File containing the private key (otherwise the private key
#'   will be taken from certfile as well).
#' @param cert_password Password to be used if the private key is encrypted and
#'   a password is necessary.
#'
#' @export
ignite_dataset <- function(
  cache_name,
  host = "localhost",
  port = 10800,
  local = FALSE,
  part = -1,
  page_size = 100,
  username = NULL,
  password = NULL,
  certfile = NULL,
  keyfile = NULL,
  cert_password = NULL) {
  dataset <- tfio_lib$ignite$IgniteDataset(
    cache_name = cache_name,
    host = host,
    port = port,
    local = local,
    part = part,
    page_size = page_size,
    username = username,
    password = password,
    certfile = certfile,
    keyfile = keyfile,
    cert_password = cert_password
  )
  as_tf_dataset(dataset)
}
