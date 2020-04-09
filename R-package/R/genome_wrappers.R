#' @title phred_sequences_to_probability
#'
#' @description Converts raw phred quality scores into base-calling error probabilities.
#'
#' @details For each ASCII encoded phred quality score (X), the probability that there
#' was an error calling that base is computed by: P = 10 ^ (-(X - 33) / 10) This is assuming an "ASCII base" of 33. The input is a tf.string tensor of ASCII encoded phred qualities,
#' one string per DNA sequence, with each character representing the quality
#' of a nucelotide. For example:
#' phred_qualities = [["BB<"], ["BBBB"]]
#'


#'
#' @return tf.RaggedTensor: The quality scores for each base in each sequence provided.
#'
#' @section For example:
#' phred_qualities = [["BB<"], ["BBBB"]]
#'
#' @export
phred_sequences_to_probability <- function() {

  python_function_result <- tf_io$genome$phred_sequences_to_probability(
)

}
#' @title read_fastq
#'
#' @description Read FastQ file into Tensor
#'
#' @details 
#'
#' @param filename Filename of the FastQ file.
#' @param name A name for the operation (optional).
#'
#' @return sequences: A string `Tensor`. raw_quality: A string `Tensor`.
#'
#' @export
read_fastq <- function(filename, name = NULL) {

  python_function_result <- tf_io$genome$read_fastq(
    filename = filename,
    name = name
  )

}
#' @title sequences_to_onehot
#'
#' @description Convert DNA sequences into a one hot nucleotide encoding.
#'
#' @details Each nucleotide in each sequence is mapped as follows:
#' A -> [1, 0, 0, 0]
#' C -> [0, 1, 0, 0]
#' G -> [0 ,0 ,1, 0]
#' T -> [0, 0, 0, 1] If for some reason a non (A, T, C, G) character exists in the string, it is
#' currently mapped to a error one hot encoding [1, 1, 1, 1].
#'


#'
#' @return tf.RaggedTensor: The output sequences with nucleotides one hot encoded.
#'
#' @section Each nucleotide in each sequence is mapped as follows:
#' A -> [1, 0, 0, 0] C -> [0, 1, 0, 0] G -> [0 ,0 ,1, 0] T -> [0, 0, 0, 1]
#'
#' @export
sequences_to_onehot <- function() {

  python_function_result <- tf_io$genome$sequences_to_onehot(
)

}
