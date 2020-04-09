#' @title decode_dicom_data
#'
#' @description Getting DICOM Tag Data.
#'
#' @details This package has two operations which wrap `DCMTK` functions.
#' `decode_dicom_image` decodes the pixel data from DICOM files, and
#' `decode_dicom_data` decodes tag information.
#' `dicom_tags` contains useful DICOM tags such as `dicom_tags.PatientsName`.
#' We borrow the same tag notation from the
#' [`pydicom`](https://pydicom.github.io/) dicom package. The detailed usage of DICOM is available in
#' [tutorial](https://www.tensorflow.org/io/tutorials/dicom). If this package helped, please kindly cite the below:
#' ```
#' @misc{marcelo_lerendegui_2019_3337331, author = {Marcelo Lerendegui and Ouwen Huang}, title = {Tensorflow Dicom Decoder}, month = jul, year = 2019, doi = {10.5281/zenodo.3337331}, url = {https://doi.org/10.5281/zenodo.3337331}
#' }
#' ```
#'
#' @param contents A Tensor of type string. 0-D. The byte string encoded DICOM file.
#' @param tags A Tensor of type `tf.uint32` of any dimension. These `uint32` numbers map directly to DICOM tags.
#' @param name A name for the operation (optional).
#'
#' @return A `Tensor` of type `tf.string` and same shape as `tags`. If a dicom tag is a list of strings, they are combined into one string and seperated by a double backslash `\`. There is a bug in [DCMTK](https://support.dcmtk.org/docs/) if the tag is a list of numbers, only the zeroth element will be returned as a string.
#'
#' @export
decode_dicom_data <- function(contents, tags = NULL, name = NULL) {

  python_function_result <- tf_io$image$decode_dicom_data(
    contents = contents,
    tags = tags,
    name = name
  )

}
#' @title decode_dicom_image
#'
#' @description Getting DICOM Image Data.
#'
#' @details This package has two operations which wrap `DCMTK` functions.
#' `decode_dicom_image` decodes the pixel data from DICOM files, and
#' `decode_dicom_data` decodes tag information.
#' `dicom_tags` contains useful DICOM tags such as `dicom_tags.PatientsName`.
#' We borrow the same tag notation from the
#' [`pydicom`](https://pydicom.github.io/) dicom package. The detailed usage of DICOM is available in
#' [tutorial](https://www.tensorflow.org/io/tutorials/dicom). If this package helped, please kindly cite the below:
#' ```
#' @misc{marcelo_lerendegui_2019_3337331, author = {Marcelo Lerendegui and Ouwen Huang}, title = {Tensorflow Dicom Decoder}, month = jul, year = 2019, doi = {10.5281/zenodo.3337331}, url = {https://doi.org/10.5281/zenodo.3337331}
#' }
#' ```
#'
#' @param contents A Tensor of type string. 0-D. The byte string encoded DICOM file.
#' @param color_dim An optional `bool`. Defaults to `FALSE`. If `TRUE`, a third channel will be appended to all images forming a 3-D tensor. A 1024 x 1024 grayscale image will be 1024 x 1024 x 1.
#' @param on_error Defaults to `skip`. This attribute establishes the behavior in case an error occurs on opening the image or if the output type cannot accomodate all the possible input values. For example if the user sets the output dtype to `tf.uint8`, but a dicom image stores a `tf.uint16` type. `strict` throws an error. `skip` returns a 1-D empty tensor. `lossy` continues with the operation scaling the value via the `scale` attribute.
#' @param scale Defaults to `preserve`. This attribute establishes what to do with the scale of the input values. `auto` will autoscale the input values, if the output type is integer, `auto` will use the maximum output scale for example a `uint8` which stores values from [0, 255] can be linearly stretched to fill a `uint16` that is [0,65535]. If the output is float, `auto` will scale to [0,1]. `preserve` keeps the values as they are, an input value greater than the maximum possible output will be clipped.
#' @param dtype An optional `tf.DType` from: `tf.uint8`, `tf.uint16`, `tf.uint32`, `tf.uint64`, `tf.float16`, `tf.float32`, `tf.float64`. Defaults to `tf.uint16`.
#' @param name A name for the operation (optional).
#'
#' @return A `Tensor` of type `dtype` and the shape is determined by the DICOM file.
#'
#' @export
decode_dicom_image <- function(contents, color_dim = FALSE, on_error = "skip", scale = "preserve", dtype = tf$uint16, name = NULL) {

  python_function_result <- tf_io$image$decode_dicom_image(
    contents = contents,
    color_dim = color_dim,
    on_error = on_error,
    scale = scale,
    dtype = dtype,
    name = name
  )

}
#' @title decode_webp
#'
#' @description Decode a WebP-encoded image to a uint8 tensor.
#'
#' @details 
#'
#' @param contents A `Tensor` of type `string`. 0-D. The WebP-encoded image.
#' @param name A name for the operation (optional).
#'
#' @return A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
#'
#' @export
decode_webp <- function(contents, name = NULL) {

  python_function_result <- tf_io$image$decode_webp(
    contents = contents,
    name = name
  )

}
#' @title encode_bmp
#'
#' @description Encode a uint8 tensor to bmp image.
#'
#' @details 
#'
#' @param image A Tensor. 3-D uint8 with shape [height, width, channels].
#' @param name A name for the operation (optional).
#'
#' @return A `Tensor` of type `string`.
#'
#' @export
encode_bmp <- function(image, name = NULL) {

  python_function_result <- tf_io$image$encode_bmp(
    image = image,
    name = name
  )

}
