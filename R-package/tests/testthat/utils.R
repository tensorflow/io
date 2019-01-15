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

iterate_all_batches <- function(dataset) {
  sess <- tf$Session()
  iterator <- make_iterator_one_shot(dataset)
  next_batch <- iterator_get_next(iterator)

  until_out_of_range({
    sess$run(next_batch)
  })
}
