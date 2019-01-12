library(tensorflow)

skip_if_no_tensorflow_io <- function() {
  if (!reticulate::py_module_available("tensorflow_io"))
    skip("tensorflow_io Python module is not available for testing")
}

test_succeeds <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_tensorflow_io()
    expect_error(force(expr), NA)
  })
}
