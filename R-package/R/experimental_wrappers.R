#' @title audio
#'
#' @description tensorflow_io.experimental.audio
#'
#' @details 
#'


#'
#' @export
 {

  python_function_result <- tf_io$experimental$audio(
)

}
#' @title ffmpeg
#'
#' @description tensorflow_io.experimental.ffmpeg
#'
#' @details 
#'


#'
#' @export
 {

  python_function_result <- tf_io$experimental$ffmpeg(
)

}
#' @title image
#'
#' @description tensorflow_io.experimental.image
#'
#' @details 
#'


#'
#' @export
 {

  python_function_result <- tf_io$experimental$image(
)

}
#' @title IODataset
#'
#' @description IODataset
#'
#' @details 
#'
#' @param function function
#' @param internal internal
#'
#' @export
IODataset <- function(function, internal = FALSE) {

  python_function_result <- tf_io$experimental$IODataset(
    function = function,
    internal = internal
  )

}
#' @title IOTensor
#'
#' @description IOTensor
#'
#' @details 
#'
#' @param spec spec
#' @param internal internal
#'
#' @export
IOTensor <- function(spec, internal = FALSE) {

  python_function_result <- tf_io$experimental$IOTensor(
    spec = spec,
    internal = internal
  )

}
#' @title serialization
#'
#' @description tensorflow_io.experimental.serialization
#'
#' @details 
#'


#'
#' @export
 {

  python_function_result <- tf_io$experimental$serialization(
)

}
#' @title text
#'
#' @description tensorflow_io.experimental.text
#'
#' @details 
#'


#'
#' @export
 {

  python_function_result <- tf_io$experimental$text(
)

}
