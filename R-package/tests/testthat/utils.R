library(tensorflow)

skip_if_no_tensorflow <- function(required_version = NULL) {
  if (!reticulate::py_module_available("tensorflow"))
    skip("TensorFlow not available for testing")
  else if (!is.null(required_version)) {
    if (tensorflow::tf_version() < required_version)
      skip(sprintf("Required version of TensorFlow (%s) not available for testing",
                   required_version))
  }
}

skip_if_no_tensorflow_io <- function(required_version = NULL) {
  if (!reticulate::py_module_available("tensorflow_io"))
    skip("TensorFlow not available for testing")
}

test_succeeds <- function(desc, expr, required_version = NULL) {
  test_that(desc, {
    skip_if_no_tensorflow(required_version)
    expect_error(force(expr), NA)
  })
}
