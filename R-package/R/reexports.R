#' Pipe operator
#'
#' See \code{\link[magrittr]{\%>\%}} for more details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
NULL

#' @importFrom tensorflow tf
#' @export
tensorflow::tf

#' @importFrom tensorflow install_tensorflow
#' @export
tensorflow::install_tensorflow

#' @importFrom tensorflow tf_config
#' @export
tensorflow::tf_config

#' @importFrom tensorflow tf_version
#' @export
tensorflow::tf_version


# Re-exports from tfdatasets dataset_iterators

#' @importFrom tfdatasets next_batch
#' @export
tfdatasets::next_batch

#' @importFrom tfdatasets with_dataset
#' @export
tfdatasets::with_dataset

#' @importFrom tfdatasets until_out_of_range
#' @export
tfdatasets::until_out_of_range

# Re-exports from tfdatasets iterators

#' @importFrom tfdatasets make_iterator_one_shot
#' @export
tfdatasets::make_iterator_one_shot

#' @importFrom tfdatasets make_iterator_initializable
#' @export
tfdatasets::make_iterator_initializable

#' @importFrom tfdatasets make_iterator_from_structure
#' @export
tfdatasets::make_iterator_from_structure

#' @importFrom tfdatasets make_iterator_from_string_handle
#' @export
tfdatasets::make_iterator_from_string_handle

#' @importFrom tfdatasets iterator_get_next
#' @export
tfdatasets::iterator_get_next

#' @importFrom tfdatasets iterator_initializer
#' @export
tfdatasets::iterator_initializer

#' @importFrom tfdatasets iterator_string_handle
#' @export
tfdatasets::iterator_string_handle

#' @importFrom tfdatasets iterator_make_initializer
#' @export
tfdatasets::iterator_make_initializer

#' @importFrom tfdatasets out_of_range_handler
#' @export
tfdatasets::out_of_range_handler


# Re-exports from tfdatasets dataset_methods

#' @importFrom tfdatasets dataset_repeat
#' @export
tfdatasets::dataset_repeat

#' @importFrom tfdatasets dataset_shuffle
#' @export
tfdatasets::dataset_shuffle

#' @importFrom tfdatasets dataset_shuffle_and_repeat
#' @export
tfdatasets::dataset_shuffle_and_repeat

#' @importFrom tfdatasets dataset_batch
#' @export
tfdatasets::dataset_batch

#' @importFrom tfdatasets dataset_cache
#' @export
tfdatasets::dataset_cache

#' @importFrom tfdatasets dataset_concatenate
#' @export
tfdatasets::dataset_concatenate

#' @importFrom tfdatasets dataset_take
#' @export
tfdatasets::dataset_take

#' @importFrom tfdatasets dataset_map
#' @export
tfdatasets::dataset_map

#' @importFrom tfdatasets dataset_map_and_batch
#' @export
tfdatasets::dataset_map_and_batch

#' @importFrom tfdatasets dataset_flat_map
#' @export
tfdatasets::dataset_flat_map

#' @importFrom tfdatasets dataset_prefetch
#' @export
tfdatasets::dataset_prefetch

#' @importFrom tfdatasets dataset_prefetch_to_device
#' @export
tfdatasets::dataset_prefetch_to_device

#' @importFrom tfdatasets dataset_filter
#' @export
tfdatasets::dataset_filter

#' @importFrom tfdatasets dataset_skip
#' @export
tfdatasets::dataset_skip

#' @importFrom tfdatasets dataset_interleave
#' @export
tfdatasets::dataset_interleave

#' @importFrom tfdatasets dataset_prefetch
#' @export
tfdatasets::dataset_prefetch

#' @importFrom tfdatasets dataset_shard
#' @export
tfdatasets::dataset_shard

#' @importFrom tfdatasets dataset_padded_batch
#' @export
tfdatasets::dataset_padded_batch

#' @importFrom tfdatasets dataset_prepare
#' @export
tfdatasets::dataset_prepare

# Re-exports from tfdatasets dataset_properties

#' @importFrom tfdatasets output_types
#' @export
tfdatasets::output_types

#' @importFrom tfdatasets output_types
#' @export
tfdatasets::output_types
