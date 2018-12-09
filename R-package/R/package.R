#' TensorFlow IO API for R
#'
#' This library provides an R interface to the
#' \href{https://github.com/tensorflow/io}{TensorFlow IO} API
#' that provides datasets and filesystem extensions maintained by SIG-IO.
#'
#' @docType package
#' @name tfio
NULL

#' @importFrom reticulate py_last_error tuple py_str py_has_attr import
#' @import tidyselect
#' @import rlang
NULL

tfio_lib <- NULL

.onLoad <- function(libname, pkgname) {

  # Delay load handler
  displayed_warning <- FALSE
  delay_load <- list(

    priority = 5,

    environment = "r-tensorflow-io",

    on_load = function() {
      check_tensorflow_version(displayed_warning)
    },

    on_error = function(e) {
      stop(tf_config()$error_message, call. = FALSE)
    }
  )

  # TODO: This is commented out for now until we add the wrappers.
  # tfio_lib <<- import("tensorflow_io", delay_load = delay_load)

}

check_tensorflow_version <- function(displayed_warning) {
  current_tf_ver <- tf_version()
  required_least_ver <- "1.12"
  if (current_tf_ver < required_least_ver) {
    if (!displayed_warning) {
      message("tfio requires TensorFlow version > ", required_least_ver, " ",
              "(you are currently running version ", current_tf_ver, ").\n")
      displayed_warning <<- TRUE
    }
  }
}

.onUnload <- function(libpath) {

}

.onAttach <- function(libname, pkgname) {

}

.onDetach <- function(libpath) {

}

# Reusable function for registering a set of methods with S3 manually. The
# methods argument is a list of character vectors, each of which has the form
# c(package, genname, class).
registerMethods <- function(methods) {
  lapply(methods, function(method) {
    pkg <- method[[1]]
    generic <- method[[2]]
    class <- method[[3]]
    func <- get(paste(generic, class, sep = "."))
    if (pkg %in% loadedNamespaces()) {
      registerS3method(generic, class, func, envir = asNamespace(pkg))
    }
    setHook(
      packageEvent(pkg, "onLoad"),
      function(...) {
        registerS3method(generic, class, func, envir = asNamespace(pkg))
      }
    )
  })
}

