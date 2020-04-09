#' @title apply
#'
#' @description Applies a transformation function to this dataset.
#'
#' @details `apply` enables chaining of custom `Dataset` transformations, which are
#' represented as functions that take one `Dataset` argument and return a
#' transformed `Dataset`. >>> dataset = tf.data.Dataset.range(100)
#' >>> def dataset_fn(ds):
#' ... return ds.filter(lambda x: x < 5)
#' >>> dataset = dataset.apply(dataset_fn)
#' >>> list(dataset.as_numpy_iterator())
#' [0, 1, 2, 3, 4]
#'
#' @param transformation_func A function that takes one `Dataset` argument and returns a `Dataset`.
#'
#' @return Dataset: The `Dataset` returned by applying `transformation_func` to this dataset.
#'
#' @export
apply <- function(transformation_func) {

  python_function_result <- tf_io$IODataset$apply(
    transformation_func = transformation_func
  )

}
#' @title batch
#'
#' @description Combines consecutive elements of this dataset into batches.
#'
#' @details >>> dataset = tf.data.Dataset.range(8)
#' >>> dataset = dataset.batch(3)
#' >>> list(dataset.as_numpy_iterator())
#' [array([0, 1, 2]), array([3, 4, 5]), array([6, 7])] >>> dataset = tf.data.Dataset.range(8)
#' >>> dataset = dataset.batch(3, drop_remainder=TRUE)
#' >>> list(dataset.as_numpy_iterator())
#' [array([0, 1, 2]), array([3, 4, 5])] The components of the resulting element will have an additional outer
#' dimension, which will be `batch_size` (or `N % batch_size` for the last
#' element if `batch_size` does not divide the number of input elements `N`
#' evenly and `drop_remainder` is `FALSE`). If your program depends on the
#' batches having the same outer dimension, you should set the `drop_remainder`
#' argument to `TRUE` to prevent the smaller batch from being produced.
#'
#' @param batch_size A `tf.int64` scalar `tf.Tensor`, representing the number of consecutive elements of this dataset to combine in a single batch.
#' @param drop_remainder (Optional.) A `tf.bool` scalar `tf.Tensor`, representing whether the last batch should be dropped in the case it has fewer than `batch_size` elements; the default behavior is not to drop the smaller batch.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
batch <- function(batch_size, drop_remainder = FALSE) {

  python_function_result <- tf_io$IODataset$batch(
    batch_size = batch_size,
    drop_remainder = drop_remainder
  )

}
#' @title cache
#'
#' @description Caches the elements in this dataset.
#'
#' @details The first time the dataset is iterated over, its elements will be cached
#' either in the specified file or in memory. Subsequent iterations will
#' use the cached data. Note: For the cache to be finalized, the input dataset must be iterated
#' through in its entirety. Otherwise, subsequent iterations will not use
#' cached data. >>> dataset = tf.data.Dataset.range(5)
#' >>> dataset = dataset.map(lambda x: x**2)
#' >>> dataset = dataset.cache()
#' >>> # The first time reading through the data will generate the data using
#' >>> # `range` and `map`.
#' >>> list(dataset.as_numpy_iterator())
#' [0, 1, 4, 9, 16]
#' >>> # Subsequent iterations read from the cache.
#' >>> list(dataset.as_numpy_iterator())
#' [0, 1, 4, 9, 16] When caching to a file, the cached data will persist across runs. Even the
#' first iteration through the data will read from the cache file. Changing
#' the input pipeline before the call to `.cache()` will have no effect until
#' the cache file is removed or the filename is changed. >>> dataset = tf.data.Dataset.range(5)
#' >>> dataset = dataset.cache("/path/to/file) # doctest: +SKIP
#' >>> list(dataset.as_numpy_iterator()) # doctest: +SKIP
#' [0, 1, 2, 3, 4]
#' >>> dataset = tf.data.Dataset.range(10)
#' >>> dataset = dataset.cache("/path/to/file") # Same file! # doctest: +SKIP
#' >>> list(dataset.as_numpy_iterator()) # doctest: +SKIP
#' [0, 1, 2, 3, 4] Note: `cache` will produce exactly the same elements during each iteration
#' through the dataset. If you wish to randomize the iteration order, make sure
#' to call `shuffle` *after* calling `cache`.
#'
#' @param filename A `tf.string` scalar `tf.Tensor`, representing the name of a directory on the filesystem to use for caching elements in this Dataset. If a filename is not provided, the dataset will be cached in memory.
#'
#' @return Dataset: A `Dataset`.
#'
#' @section Note: For the cache to be finalized, the input dataset must be iterated:
#' through in its entirety. Otherwise, subsequent iterations will not use cached data.
#'
#' @section Note: `cache` will produce exactly the same elements during each iteration:
#' through the dataset. If you wish to randomize the iteration order, make sure to call `shuffle` *after* calling `cache`.
#'
#' @export
cache <- function(filename = "") {

  python_function_result <- tf_io$IODataset$cache(
    filename = filename
  )

}
#' @title concatenate
#'
#' @description Creates a `Dataset` by concatenating the given dataset with this dataset.
#'
#' @details >>> a = tf.data.Dataset.range(1, 4) # ==> [ 1, 2, 3 ]
#' >>> b = tf.data.Dataset.range(4, 8) # ==> [ 4, 5, 6, 7 ]
#' >>> ds = a.concatenate(b)
#' >>> list(ds.as_numpy_iterator())
#' [1, 2, 3, 4, 5, 6, 7]
#' >>> # The input dataset and dataset to be concatenated should have the same
#' >>> # nested structures and output types.
#' >>> c = tf.data.Dataset.zip((a, b))
#' >>> a.concatenate(c)
#' Traceback (most recent call last):
#' TypeError: Two datasets to concatenate have different types
#' <dtype: 'int64'> and (tf.int64, tf.int64)
#' >>> d = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
#' >>> a.concatenate(d)
#' Traceback (most recent call last):
#' TypeError: Two datasets to concatenate have different types
#' <dtype: 'int64'> and <dtype: 'string'>
#'
#' @param dataset `Dataset` to be concatenated.
#'
#' @return Dataset: A `Dataset`.
#'
#' @section TypeError: Two datasets to concatenate have different types:
#' <dtype: 'int64'> and <dtype: 'string'>
#'
#' @export
concatenate <- function(dataset) {

  python_function_result <- tf_io$IODataset$concatenate(
    dataset = dataset
  )

}
#' @title element_spec
#'
#' @description The type specification of an element of this dataset.
#'
#' @details >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3]).element_spec
#' TensorSpec(shape=(), dtype=tf.int32, name=NULL) Returns: A nested structure of `tf.TypeSpec` objects matching the structure of an element of this dataset and specifying the type of individual components.
#'


#'
#' @return A nested structure of `tf.TypeSpec` objects matching the structure of an element of this dataset and specifying the type of individual components.
#'
#' @export
 {

  python_function_result <- tf_io$IODataset$element_spec(
)

}
#' @title enumerate
#'
#' @description Enumerates the elements of this dataset.
#'
#' @details It is similar to python's `enumerate`. >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
#' >>> dataset = dataset.enumerate(start=5)
#' >>> for element in dataset.as_numpy_iterator():
#' ... print(element)
#' (5, 1)
#' (6, 2)
#' (7, 3) >>> # The nested structure of the input dataset determines the structure of
#' >>> # elements in the resulting dataset.
#' >>> dataset = tf.data.Dataset.from_tensor_slices([(7, 8), (9, 10)])
#' >>> dataset = dataset.enumerate()
#' >>> for element in dataset.as_numpy_iterator():
#' ... print(element)
#' (0, array([7, 8], dtype=int32))
#' (1, array([ 9, 10], dtype=int32))
#'
#' @param start A `tf.int64` scalar `tf.Tensor`, representing the start value for enumeration.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
enumerate <- function(start = 0L) {

  python_function_result <- tf_io$IODataset$enumerate(
    start = start
  )

}
#' @title filter
#'
#' @description Filters this dataset according to `predicate`.
#'
#' @details >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
#' >>> dataset = dataset.filter(lambda x: x < 3)
#' >>> list(dataset.as_numpy_iterator())
#' [1, 2]
#' >>> # `tf.math.equal(x, y)` is required for equality comparison
#' >>> def filter_fn(x):
#' ... return tf.math.equal(x, 1)
#' >>> dataset = dataset.filter(filter_fn)
#' >>> list(dataset.as_numpy_iterator())
#' [1]
#'
#' @param predicate A function mapping a dataset element to a boolean.
#'
#' @return Dataset: The `Dataset` containing the elements of this dataset for which `predicate` is `TRUE`.
#'
#' @export
filter <- function(predicate) {

  python_function_result <- tf_io$IODataset$filter(
    predicate = predicate
  )

}
#' @title flat_map
#'
#' @description Maps `map_func` across this dataset and flattens the result.
#'
#' @details Use `flat_map` if you want to make sure that the order of your dataset
#' stays the same. For example, to flatten a dataset of batches into a
#' dataset of their elements: >>> dataset = Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#' >>> dataset = dataset.flat_map(lambda x: Dataset.from_tensor_slices(x))
#' >>> list(dataset.as_numpy_iterator())
#' [1, 2, 3, 4, 5, 6, 7, 8, 9] `tf.data.Dataset.interleave()` is a generalization of `flat_map`, since
#' `flat_map` produces the same output as
#' `tf.data.Dataset.interleave(cycle_length=1)`
#'
#' @param map_func A function mapping a dataset element to a dataset.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
flat_map <- function(map_func) {

  python_function_result <- tf_io$IODataset$flat_map(
    map_func = map_func
  )

}
#' @title from_audio
#'
#' @description Creates an `IODataset` from an audio file.
#'
#' @details The following audio file formats are supported:
#' - WAV
#'
#' @param cls cls
#' @param filename A string, the filename of an audio file.
#'
#' @return A `IODataset`.
#'
#' @section The following audio file formats are supported:
#' - WAV
#'
#' @export
from_audio <- function(cls, filename) {

  python_function_result <- tf_io$IODataset$from_audio(
    cls = cls,
    filename = filename
  )

}
#' @title from_avro
#'
#' @description Creates an `IODataset` from a avro file's dataset object.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a avro file.
#' @param schema A string, the schema of a avro file.
#' @param columns A list of column names within avro file.
#'
#' @return A `IODataset`.
#'
#' @export
from_avro <- function(cls, filename, schema, columns = NULL) {

  python_function_result <- tf_io$IODataset$from_avro(
    cls = cls,
    filename = filename,
    schema = schema,
    columns = columns
  )

}
#' @title from_ffmpeg
#'
#' @description Creates an `IODataset` from a media file by FFmpeg
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a media file.
#' @param stream A string, the stream index (e.g., "v:0"). Note video, audio, and subtitle index starts with 0 separately.
#'
#' @return A `IODataset`.
#'
#' @export
from_ffmpeg <- function(cls, filename, stream) {

  python_function_result <- tf_io$IODataset$from_ffmpeg(
    cls = cls,
    filename = filename,
    stream = stream
  )

}
#' @title from_generator
#'
#' @description Creates a `Dataset` whose elements are generated by `generator`.
#'
#' @details The `generator` argument must be a callable object that returns
#' an object that supports the `iter()` protocol (e.g. a generator function).
#' The elements generated by `generator` must be compatible with the given
#' `output_types` and (optional) `output_shapes` arguments. >>> import itertools
#' >>>
#' >>> def gen():
#' ... for i in itertools.count(1):
#' ... yield (i, [1] * i)
#' >>>
#' >>> dataset = tf.data.Dataset.from_generator(
#' ... gen,
#' ... (tf.int64, tf.int64),
#' ... (tf.TensorShape([]), tf.TensorShape([NULL])))
#' >>>
#' >>> list(dataset.take(3).as_numpy_iterator())
#' [(1, array([1])), (2, array([1, 1])), (3, array([1, 1, 1]))] NOTE: The current implementation of `Dataset.from_generator()` uses
#' `tf.numpy_function` and inherits the same constraints. In particular, it
#' requires the `Dataset`- and `Iterator`-related operations to be placed
#' on a device in the same process as the Python program that called
#' `Dataset.from_generator()`. The body of `generator` will not be
#' serialized in a `GraphDef`, and you should not use this method if you
#' need to serialize your model and restore it in a different environment. NOTE: If `generator` depends on mutable global variables or other external
#' state, be aware that the runtime may invoke `generator` multiple times
#' (in order to support repeating the `Dataset`) and at any time
#' between the call to `Dataset.from_generator()` and the production of the
#' first element from the generator. Mutating global variables or external
#' state can cause undefined behavior, and we recommend that you explicitly
#' cache any external state in `generator` before calling
#' `Dataset.from_generator()`.
#'
#' @param generator A callable object that returns an object that supports the `iter()` protocol. If `args` is not specified, `generator` must take no arguments; otherwise it must take as many arguments as there are values in `args`.
#' @param output_types A nested structure of `tf.DType` objects corresponding to each component of an element yielded by `generator`.
#' @param output_shapes (Optional.) A nested structure of `tf.TensorShape` objects corresponding to each component of an element yielded by `generator`.
#' @param args (Optional.) A list of `tf.Tensor` objects that will be evaluated and passed to `generator` as NumPy-array arguments.
#'
#' @return Dataset: A `Dataset`.
#'
#' @section NOTE: The current implementation of `Dataset.from_generator()` uses:
#' `tf.numpy_function` and inherits the same constraints. In particular, it requires the `Dataset`- and `Iterator`-related operations to be placed on a device in the same process as the Python program that called `Dataset.from_generator()`. The body of `generator` will not be serialized in a `GraphDef`, and you should not use this method if you need to serialize your model and restore it in a different environment.
#'
#' @section NOTE: If `generator` depends on mutable global variables or other external:
#' state, be aware that the runtime may invoke `generator` multiple times (in order to support repeating the `Dataset`) and at any time between the call to `Dataset.from_generator()` and the production of the first element from the generator. Mutating global variables or external state can cause undefined behavior, and we recommend that you explicitly cache any external state in `generator` before calling `Dataset.from_generator()`.
#'
#' @export
from_generator <- function(generator, output_types, output_shapes = NULL, args = NULL) {

  python_function_result <- tf_io$IODataset$from_generator(
    generator = generator,
    output_types = output_types,
    output_shapes = output_shapes,
    args = args
  )

}
#' @title from_hdf5
#'
#' @description Creates an `IODataset` from a hdf5 file's dataset object.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a hdf5 file.
#' @param dataset A string, the dataset name within hdf5 file.
#' @param spec A tf.TensorSpec or a dtype (e.g., tf.int64) of the dataset. In graph mode, spec is needed. In eager mode, spec is probed automatically.
#'
#' @return A `IODataset`.
#'
#' @export
from_hdf5 <- function(cls, filename, dataset, spec = NULL) {

  python_function_result <- tf_io$IODataset$from_hdf5(
    cls = cls,
    filename = filename,
    dataset = dataset,
    spec = spec
  )

}
#' @title from_json
#'
#' @description Creates an `IODataset` from a json file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a json file.
#' @param columns A list of column names. By default (NULL) all columns will be read.
#' @param mode A string, the mode (records or NULL) to open json file.
#'
#' @return A `IODataset`.
#'
#' @export
from_json <- function(cls, filename, columns = NULL, mode = NULL) {

  python_function_result <- tf_io$IODataset$from_json(
    cls = cls,
    filename = filename,
    columns = columns,
    mode = mode
  )

}
#' @title from_kafka
#'
#' @description Creates an `IODataset` from kafka server with an offset range.
#'
#' @details 
#'
#' @param cls cls
#' @param topic A `tf.string` tensor containing topic subscription.
#' @param partition A `tf.int64` tensor containing the partition, by default 0.
#' @param start A `tf.int64` tensor containing the start offset, by default 0.
#' @param stop A `tf.int64` tensor containing the end offset, by default -1.
#' @param servers An optional list of bootstrap servers, by default `localhost:9092`.
#' @param configuration An optional `tf.string` tensor containing configurations in [Key=Value] format. There are three types of configurations: Global configuration: please refer to 'Global configuration properties' in librdkafka doc. Examples include ["enable.auto.commit=false", "heartbeat.interval.ms=2000"] Topic configuration: please refer to 'Topic configuration properties' in librdkafka doc. Note all topic configurations should be prefixed with `configuration.topic.`. Examples include ["conf.topic.auto.offset.reset=earliest"]
#'
#' @return A `IODataset`.
#'
#' @export
from_kafka <- function(cls, topic, partition = 0L, start = 0L, stop = -1L, servers = NULL, configuration = NULL) {

  python_function_result <- tf_io$IODataset$from_kafka(
    cls = cls,
    topic = topic,
    partition = partition,
    start = start,
    stop = stop,
    servers = servers,
    configuration = configuration
  )

}
#' @title from_lmdb
#'
#' @description Creates an `IODataset` from a lmdb file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a lmdb file.
#'
#' @return A `IODataset`.
#'
#' @export
from_lmdb <- function(cls, filename) {

  python_function_result <- tf_io$IODataset$from_lmdb(
    cls = cls,
    filename = filename
  )

}
#' @title from_mnist
#'
#' @description Creates an `IODataset` from MNIST images and/or labels files.
#'
#' @details 
#'
#' @param cls cls
#' @param images A string, the filename of MNIST images file.
#' @param labels A string, the filename of MNIST labels file.
#'
#' @return A `IODataset`.
#'
#' @export
from_mnist <- function(cls, images = NULL, labels = NULL) {

  python_function_result <- tf_io$IODataset$from_mnist(
    cls = cls,
    images = images,
    labels = labels
  )

}
#' @title from_parquet
#'
#' @description Creates an `IODataset` from a json file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a json file.
#' @param columns A list of column names. By default (NULL) all columns will be read.
#'
#' @return A `IODataset`.
#'
#' @export
from_parquet <- function(cls, filename, columns = NULL) {

  python_function_result <- tf_io$IODataset$from_parquet(
    cls = cls,
    filename = filename,
    columns = columns
  )

}
#' @title from_pcap
#'
#' @description Creates an `IODataset` from a pcap file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a pcap file.
#'
#' @return A `IODataset`.
#'
#' @export
from_pcap <- function(cls, filename) {

  python_function_result <- tf_io$IODataset$from_pcap(
    cls = cls,
    filename = filename
  )

}
#' @title from_tensor_slices
#'
#' @description Creates a `Dataset` whose elements are slices of the given tensors.
#'
#' @details The given tensors are sliced along their first dimension. This operation
#' preserves the structure of the input tensors, removing the first dimension
#' of each tensor and using it as the dataset dimension. All input tensors
#' must have the same size in their first dimensions. >>> # Slicing a 1D tensor produces scalar tensor elements.
#' >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
#' >>> list(dataset.as_numpy_iterator())
#' [1, 2, 3] >>> # Slicing a 2D tensor produces 1D tensor elements.
#' >>> dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
#' >>> list(dataset.as_numpy_iterator())
#' [array([1, 2], dtype=int32), array([3, 4], dtype=int32)] >>> # Slicing a list of 1D tensors produces list elements containing
#' >>> # scalar tensors.
#' >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
#' >>> list(dataset.as_numpy_iterator())
#' [(1, 3, 5), (2, 4, 6)] >>> # Dictionary structure is also preserved.
#' >>> dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
#' >>> list(dataset.as_numpy_iterator()) == [{'a': 1, 'b': 3},
#' ... {'a': 2, 'b': 4}]
#' TRUE >>> # Two tensors can be combined into one Dataset object.
#' >>> features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
#' >>> labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
#' >>> dataset = Dataset.from_tensor_slices((features, labels))
#' >>> # Both the features and the labels tensors can be converted
#' >>> # to a Dataset object separately and combined after.
#' >>> features_dataset = Dataset.from_tensor_slices(features)
#' >>> labels_dataset = Dataset.from_tensor_slices(labels)
#' >>> dataset = Dataset.zip((features_dataset, labels_dataset))
#' >>> # A batched feature and label set can be converted to a Dataset
#' >>> # in similar fashion.
#' >>> batched_features = tf.constant([[[1, 3], [2, 3]],
#' ... [[2, 1], [1, 2]],
#' ... [[3, 3], [3, 2]]], shape=(3, 2, 2))
#' >>> batched_labels = tf.constant([['A', 'A'],
#' ... ['B', 'B'],
#' ... ['A', 'B']], shape=(3, 2, 1))
#' >>> dataset = Dataset.from_tensor_slices((batched_features, batched_labels))
#' >>> for element in dataset.as_numpy_iterator():
#' ... print(element)
#' (array([[1, 3], [2, 3]], dtype=int32), array([[b'A'], [b'A']], dtype=object))
#' (array([[2, 1], [1, 2]], dtype=int32), array([[b'B'], [b'B']], dtype=object))
#' (array([[3, 3], [3, 2]], dtype=int32), array([[b'A'], [b'B']], dtype=object)) Note that if `tensors` contains a NumPy array, and eager execution is not
#' enabled, the values will be embedded in the graph as one or more
#' `tf.constant` operations. For large datasets (> 1 GB), this can waste
#' memory and run into byte limits of graph serialization. If `tensors`
#' contains one or more large NumPy arrays, consider the alternative described
#' in [this guide](
#' https://tensorflow.org/guide/data#consuming_numpy_arrays).
#'
#' @param tensors A dataset element, with each component having the same size in the first dimension.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
from_tensor_slices <- function(tensors) {

  python_function_result <- tf_io$IODataset$from_tensor_slices(
    tensors = tensors
  )

}
#' @title from_tensors
#'
#' @description Creates a `Dataset` with a single element, comprising the given tensors.
#'
#' @details >>> dataset = tf.data.Dataset.from_tensors([1, 2, 3])
#' >>> list(dataset.as_numpy_iterator())
#' [array([1, 2, 3], dtype=int32)]
#' >>> dataset = tf.data.Dataset.from_tensors(([1, 2, 3], 'A'))
#' >>> list(dataset.as_numpy_iterator())
#' [(array([1, 2, 3], dtype=int32), b'A')] Note that if `tensors` contains a NumPy array, and eager execution is not
#' enabled, the values will be embedded in the graph as one or more
#' `tf.constant` operations. For large datasets (> 1 GB), this can waste
#' memory and run into byte limits of graph serialization. If `tensors`
#' contains one or more large NumPy arrays, consider the alternative described
#' in [this
#' guide](https://tensorflow.org/guide/data#consuming_numpy_arrays).
#'
#' @param tensors A dataset element.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
from_tensors <- function(tensors) {

  python_function_result <- tf_io$IODataset$from_tensors(
    tensors = tensors
  )

}
#' @title graph
#'
#' @description Obtain a GraphIODataset to be used in graph mode.
#'
#' @details 
#'
#' @param cls cls
#' @param dtype Data type of the GraphIODataset.
#'
#' @return A class of `GraphIODataset`.
#'
#' @export
graph <- function(cls, dtype) {

  python_function_result <- tf_io$IODataset$graph(
    cls = cls,
    dtype = dtype
  )

}
#' @title interleave
#'
#' @description Maps `map_func` across this dataset, and interleaves the results.
#'
#' @details For example, you can use `Dataset.interleave()` to process many input files
#' concurrently: >>> # Preprocess 4 files concurrently, and interleave blocks of 16 records
#' >>> # from each file.
#' >>> filenames = ["/var/data/file1.txt", "/var/data/file2.txt",
#' ... "/var/data/file3.txt", "/var/data/file4.txt"]
#' >>> dataset = tf.data.Dataset.from_tensor_slices(filenames)
#' >>> def parse_fn(filename):
#' ... return tf.data.Dataset.range(10)
#' >>> dataset = dataset.interleave(lambda x:
#' ... tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
#' ... cycle_length=4, block_length=16) The `cycle_length` and `block_length` arguments control the order in which
#' elements are produced. `cycle_length` controls the number of input elements
#' that are processed concurrently. If you set `cycle_length` to 1, this
#' transformation will handle one input element at a time, and will produce
#' identical results to `tf.data.Dataset.flat_map`. In general,
#' this transformation will apply `map_func` to `cycle_length` input elements,
#' open iterators on the returned `Dataset` objects, and cycle through them
#' producing `block_length` consecutive elements from each iterator, and
#' consuming the next input element each time it reaches the end of an
#' iterator. For example: >>> dataset = Dataset.range(1, 6) # ==> [ 1, 2, 3, 4, 5 ]
#' >>> # NOTE: New lines indicate "block" boundaries.
#' >>> dataset = dataset.interleave(
#' ... lambda x: Dataset.from_tensors(x).repeat(6),
#' ... cycle_length=2, block_length=4)
#' >>> list(dataset.as_numpy_iterator())
#' [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5] NOTE: The order of elements yielded by this transformation is
#' deterministic, as long as `map_func` is a pure function. If
#' `map_func` contains any stateful operations, the order in which
#' that state is accessed is undefined.
#'
#' @param map_func A function mapping a dataset element to a dataset.
#' @param cycle_length (Optional.) The number of input elements that will be processed concurrently. If not specified, the value will be derived from the number of available CPU cores. If the `num_parallel_calls` argument is set to `tf.data.experimental.AUTOTUNE`, the `cycle_length` argument also identifies the maximum degree of parallelism.
#' @param block_length (Optional.) The number of consecutive elements to produce from each input element before cycling to another input element.
#' @param num_parallel_calls (Optional.) If specified, the implementation creates a threadpool, which is used to fetch inputs from cycle elements asynchronously and in parallel. The default behavior is to fetch inputs from cycle elements synchronously with no parallelism. If the value `tf.data.experimental.AUTOTUNE` is used, then the number of parallel calls is set dynamically based on available CPU.
#'
#' @return Dataset: A `Dataset`.
#'
#' @section NOTE: The order of elements yielded by this transformation is:
#' deterministic, as long as `map_func` is a pure function. If `map_func` contains any stateful operations, the order in which that state is accessed is undefined.
#'
#' @export
interleave <- function(map_func, cycle_length = -1L, block_length = 1L, num_parallel_calls = NULL) {

  python_function_result <- tf_io$IODataset$interleave(
    map_func = map_func,
    cycle_length = cycle_length,
    block_length = block_length,
    num_parallel_calls = num_parallel_calls
  )

}
#' @title list_files
#'
#' @description A dataset of all files matching one or more glob patterns.
#'
#' @details The `file_pattern` argument should be a small number of glob patterns.
#' If your filenames have already been globbed, use
#' `Dataset.from_tensor_slices(filenames)` instead, as re-globbing every
#' filename with `list_files` may result in poor performance with remote
#' storage systems. NOTE: The default behavior of this method is to return filenames in
#' a non-deterministic random shuffled order. Pass a `seed` or `shuffle=FALSE`
#' to get results in a deterministic order. Example: If we had the following files on our filesystem: - /path/to/dir/a.txt - /path/to/dir/b.py - /path/to/dir/c.py If we pass "/path/to/dir/*.py" as the directory, the dataset would produce: - /path/to/dir/b.py - /path/to/dir/c.py
#'
#' @param file_pattern A string, a list of strings, or a `tf.Tensor` of string type (scalar or vector), representing the filename glob (i.e. shell wildcard) pattern(s) that will be matched.
#' @param shuffle (Optional.) If `TRUE`, the file names will be shuffled randomly. Defaults to `TRUE`.
#' @param seed (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random seed that will be used to create the distribution. See `tf.compat.v1.set_random_seed` for behavior.
#'
#' @return Dataset: A `Dataset` of strings corresponding to file names.
#'
#' @section NOTE: The default behavior of this method is to return filenames in:
#' a non-deterministic random shuffled order. Pass a `seed` or `shuffle=FALSE` to get results in a deterministic order.
#'
#' @section Example:
#' If we had the following files on our filesystem: - /path/to/dir/a.txt - /path/to/dir/b.py - /path/to/dir/c.py If we pass "/path/to/dir/*.py" as the directory, the dataset would produce: - /path/to/dir/b.py - /path/to/dir/c.py
#'
#' @export
list_files <- function(file_pattern, shuffle = NULL, seed = NULL) {

  python_function_result <- tf_io$IODataset$list_files(
    file_pattern = file_pattern,
    shuffle = shuffle,
    seed = seed
  )

}
#' @title map
#'
#' @description Maps `map_func` across the elements of this dataset.
#'
#' @details This transformation applies `map_func` to each element of this dataset, and
#' returns a new dataset containing the transformed elements, in the same
#' order as they appeared in the input. `map_func` can be used to change both
#' the values and the structure of a dataset's elements. For example, adding 1
#' to each element, or projecting a subset of element components. >>> dataset = Dataset.range(1, 6) # ==> [ 1, 2, 3, 4, 5 ]
#' >>> dataset = dataset.map(lambda x: x + 1)
#' >>> list(dataset.as_numpy_iterator())
#' [2, 3, 4, 5, 6] The input signature of `map_func` is determined by the structure of each
#' element in this dataset. >>> dataset = Dataset.range(5)
#' >>> # `map_func` takes a single argument of type `tf.Tensor` with the same
#' >>> # shape and dtype.
#' >>> result = dataset.map(lambda x: x + 1) >>> # Each element is a list containing two `tf.Tensor` objects.
#' >>> elements = [(1, "foo"), (2, "bar"), (3, "baz)")]
#' >>> dataset = tf.data.Dataset.from_generator(
#' ... lambda: elements, (tf.int32, tf.string))
#' >>> # `map_func` takes two arguments of type `tf.Tensor`. This function
#' >>> # projects out just the first component.
#' >>> result = dataset.map(lambda x_int, y_str: x_int)
#' >>> list(result.as_numpy_iterator())
#' [1, 2, 3] >>> # Each element is a dictionary mapping strings to `tf.Tensor` objects.
#' >>> elements = ([{"a": 1, "b": "foo"},
#' ... {"a": 2, "b": "bar"},
#' ... {"a": 3, "b": "baz"}])
#' >>> dataset = tf.data.Dataset.from_generator(
#' ... lambda: elements, {"a": tf.int32, "b": tf.string})
#' >>> # `map_func` takes a single argument of type `dict` with the same keys
#' >>> # as the elements.
#' >>> result = dataset.map(lambda d: str(d["a"]) + d["b"]) The value or values returned by `map_func` determine the structure of each
#' element in the returned dataset. >>> dataset = tf.data.Dataset.range(3)
#' >>> # `map_func` returns two `tf.Tensor` objects.
#' >>> def g(x):
#' ... return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"])
#' >>> result = dataset.map(g)
#' >>> result.element_spec
#' (TensorSpec(shape=(), dtype=tf.float32, name=NULL), TensorSpec(shape=(3,), dtype=tf.string, name=NULL))
#' >>> # Python primitives, lists, and NumPy arrays are implicitly converted to
#' >>> # `tf.Tensor`.
#' >>> def h(x):
#' ... return 37.0, ["Foo", "Bar"], np.array([1.0, 2.0], dtype=np.float64)
#' >>> result = dataset.map(h)
#' >>> result.element_spec
#' (TensorSpec(shape=(), dtype=tf.float32, name=NULL), TensorSpec(shape=(2,), dtype=tf.string, name=NULL), TensorSpec(shape=(2,), dtype=tf.float64, name=NULL))
#' >>> # `map_func` can return nested structures.
#' >>> def i(x):
#' ... return (37.0, [42, 16]), "foo"
#' >>> result = dataset.map(i)
#' >>> result.element_spec
#' ((TensorSpec(shape=(), dtype=tf.float32, name=NULL), TensorSpec(shape=(2,), dtype=tf.int32, name=NULL)), TensorSpec(shape=(), dtype=tf.string, name=NULL)) `map_func` can accept as arguments and return any type of dataset element. Note that irrespective of the context in which `map_func` is defined (eager
#' vs. graph), tf.data traces the function and executes it as a graph. To use
#' Python code inside of the function you have two options: 1) Rely on AutoGraph to convert Python code into an equivalent graph
#' computation. The downside of this approach is that AutoGraph can convert
#' some but not all Python code. 2) Use `tf.py_function`, which allows you to write arbitrary Python code but
#' will generally result in worse performance than 1). For example: >>> d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
#' >>> # transform a string tensor to upper case string using a Python function
#' >>> def upper_case_fn(t: tf.Tensor):
#' ... return t.numpy().decode('utf-8').upper()
#' >>> d = d.map(lambda x: tf.py_function(func=upper_case_fn,
#' ... inp=[x], Tout=tf.string))
#' >>> list(d.as_numpy_iterator())
#' [b'HELLO', b'WORLD']
#'
#' @param map_func A function mapping a dataset element to another dataset element.
#' @param num_parallel_calls (Optional.) A `tf.int32` scalar `tf.Tensor`, representing the number elements to process asynchronously in parallel. If not specified, elements will be processed sequentially. If the value `tf.data.experimental.AUTOTUNE` is used, then the number of parallel calls is set dynamically based on available CPU.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
map <- function(map_func, num_parallel_calls = NULL) {

  python_function_result <- tf_io$IODataset$map(
    map_func = map_func,
    num_parallel_calls = num_parallel_calls
  )

}
#' @title padded_batch
#'
#' @description Combines consecutive elements of this dataset into padded batches.
#'
#' @details This transformation combines multiple consecutive elements of the input
#' dataset into a single element. Like `tf.data.Dataset.batch`, the components of the resulting element will
#' have an additional outer dimension, which will be `batch_size` (or
#' `N % batch_size` for the last element if `batch_size` does not divide the
#' number of input elements `N` evenly and `drop_remainder` is `FALSE`). If
#' your program depends on the batches having the same outer dimension, you
#' should set the `drop_remainder` argument to `TRUE` to prevent the smaller
#' batch from being produced. Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
#' different shapes, and this transformation will pad each component to the
#' respective shape in `padding_shapes`. The `padding_shapes` argument
#' determines the resulting shape for each dimension of each component in an
#' output element: * If the dimension is a constant (e.g. `tf.compat.v1.Dimension(37)`), the
#' component will be padded out to that length in that dimension.
#' * If the dimension is unknown (e.g. `tf.compat.v1.Dimension(NULL)`), the
#' component will be padded out to the maximum length of all elements in that
#' dimension. >>> elements = [[1, 2],
#' ... [3, 4, 5],
#' ... [6, 7],
#' ... [8]]
#' >>> A = tf.data.Dataset.from_generator(lambda: iter(elements), tf.int32)
#' >>> # Pad to the smallest per-batch size that fits all elements.
#' >>> B = A.padded_batch(2, padded_shapes=[NULL])
#' >>> for element in B.as_numpy_iterator():
#' ... print(element)
#' [[1 2 0] [3 4 5]]
#' [[6 7] [8 0]]
#' >>> # Pad to a fixed size.
#' >>> C = A.padded_batch(2, padded_shapes=3)
#' >>> for element in C.as_numpy_iterator():
#' ... print(element)
#' [[1 2 0] [3 4 5]]
#' [[6 7 0] [8 0 0]]
#' >>> # Pad with a custom value.
#' >>> D = A.padded_batch(2, padded_shapes=3, padding_values=-1)
#' >>> for element in D.as_numpy_iterator():
#' ... print(element)
#' [[ 1 2 -1] [ 3 4 5]]
#' [[ 6 7 -1] [ 8 -1 -1]]
#' >>> # Components of nested elements can be padded independently.
#' >>> elements = [([1, 2, 3], [10]),
#' ... ([4, 5], [11, 12])]
#' >>> dataset = tf.data.Dataset.from_generator(
#' ... lambda: iter(elements), (tf.int32, tf.int32))
#' >>> # Pad the first component of the list to length 4, and the second
#' >>> # component to the smallest size that fits.
#' >>> dataset = dataset.padded_batch(2,
#' ... padded_shapes=([4], [NULL]),
#' ... padding_values=(-1, 100))
#' >>> list(dataset.as_numpy_iterator())
#' [(array([[ 1, 2, 3, -1], [ 4, 5, -1, -1]], dtype=int32), array([[ 10, 100], [ 11, 12]], dtype=int32))] See also `tf.data.experimental.dense_to_sparse_batch`, which combines
#' elements that may have different shapes into a `tf.SparseTensor`.
#'
#' @param batch_size A `tf.int64` scalar `tf.Tensor`, representing the number of consecutive elements of this dataset to combine in a single batch.
#' @param padded_shapes A nested structure of `tf.TensorShape` or `tf.int64` vector tensor-like objects representing the shape to which the respective component of each input element should be padded prior to batching. Any unknown dimensions (e.g. `tf.compat.v1.Dimension(NULL)` in a `tf.TensorShape` or `-1` in a tensor-like object) will be padded to the maximum size of that dimension in each batch.
#' @param padding_values (Optional.) A nested structure of scalar-shaped `tf.Tensor`, representing the padding values to use for the respective components. Defaults are `0` for numeric types and the empty string for string types.
#' @param drop_remainder (Optional.) A `tf.bool` scalar `tf.Tensor`, representing whether the last batch should be dropped in the case it has fewer than `batch_size` elements; the default behavior is not to drop the smaller batch.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
padded_batch <- function(batch_size, padded_shapes, padding_values = NULL, drop_remainder = FALSE) {

  python_function_result <- tf_io$IODataset$padded_batch(
    batch_size = batch_size,
    padded_shapes = padded_shapes,
    padding_values = padding_values,
    drop_remainder = drop_remainder
  )

}
#' @title prefetch
#'
#' @description Creates a `Dataset` that prefetches elements from this dataset.
#'
#' @details Most dataset input pipelines should end with a call to `prefetch`. This
#' allows later elements to be prepared while the current element is being
#' processed. This often improves latency and throughput, at the cost of
#' using additional memory to store prefetched elements. Note: Like other `Dataset` methods, prefetch operates on the
#' elements of the input dataset. It has no concept of examples vs. batches.
#' `examples.prefetch(2)` will prefetch two elements (2 examples),
#' while `examples.batch(20).prefetch(2)` will prefetch 2 elements
#' (2 batches, of 20 examples each). >>> dataset = tf.data.Dataset.range(3)
#' >>> dataset = dataset.prefetch(2)
#' >>> list(dataset.as_numpy_iterator())
#' [0, 1, 2]
#'
#' @param buffer_size A `tf.int64` scalar `tf.Tensor`, representing the maximum number of elements that will be buffered when prefetching.
#'
#' @return Dataset: A `Dataset`.
#'
#' @section Note: Like other `Dataset` methods, prefetch operates on the:
#' elements of the input dataset. It has no concept of examples vs. batches. `examples.prefetch(2)` will prefetch two elements (2 examples), while `examples.batch(20).prefetch(2)` will prefetch 2 elements (2 batches, of 20 examples each).
#'
#' @export
prefetch <- function(buffer_size) {

  python_function_result <- tf_io$IODataset$prefetch(
    buffer_size = buffer_size
  )

}
#' @title reduce
#'
#' @description Reduces the input dataset to a single element.
#'
#' @details The transformation calls `reduce_func` successively on every element of
#' the input dataset until the dataset is exhausted, aggregating information in
#' its internal state. The `initial_state` argument is used for the initial
#' state and the final state is returned as the result. >>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1).numpy()
#' 5
#' >>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y).numpy()
#' 10
#'
#' @param initial_state An element representing the initial state of the transformation.
#' @param reduce_func A function that maps `(old_state, input_element)` to `new_state`. It must take two arguments and return a new element The structure of `new_state` must match the structure of `initial_state`.
#'
#' @return A dataset element corresponding to the final state of the transformation.
#'
#' @export
reduce <- function(initial_state, reduce_func) {

  python_function_result <- tf_io$IODataset$reduce(
    initial_state = initial_state,
    reduce_func = reduce_func
  )

}
#' @title shard
#'
#' @description Creates a `Dataset` that includes only 1/`num_shards` of this dataset.
#'
#' @details `shard` is deterministic. The Dataset produced by `A.shard(n, i)` will
#' contain all elements of A whose index mod n = i. >>> A = tf.data.Dataset.range(10)
#' >>> B = A.shard(num_shards=3, index=0)
#' >>> list(B.as_numpy_iterator())
#' [0, 3, 6, 9]
#' >>> C = A.shard(num_shards=3, index=1)
#' >>> list(C.as_numpy_iterator())
#' [1, 4, 7]
#' >>> D = A.shard(num_shards=3, index=2)
#' >>> list(D.as_numpy_iterator())
#' [2, 5, 8] This dataset operator is very useful when running distributed training, as
#' it allows each worker to read a unique subset. When reading a single input file, you can shard elements as follows: ```python
#' d = tf.data.TFRecordDataset(input_file)
#' d = d.shard(num_workers, worker_index)
#' d = d.repeat(num_epochs)
#' d = d.shuffle(shuffle_buffer_size)
#' d = d.map(parser_fn, num_parallel_calls=num_map_threads)
#' ``` Important caveats: - Be sure to shard before you use any randomizing operator (such as shuffle).
#' - Generally it is best if the shard operator is used early in the dataset pipeline. For example, when reading from a set of TFRecord files, shard before converting the dataset to input samples. This avoids reading every file on every worker. The following is an example of an efficient sharding strategy within a complete pipeline: ```python
#' d = Dataset.list_files(pattern)
#' d = d.shard(num_workers, worker_index)
#' d = d.repeat(num_epochs)
#' d = d.shuffle(shuffle_buffer_size)
#' d = d.interleave(tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1)
#' d = d.map(parser_fn, num_parallel_calls=num_map_threads)
#' ```
#'
#' @param num_shards A `tf.int64` scalar `tf.Tensor`, representing the number of shards operating in parallel.
#' @param index A `tf.int64` scalar `tf.Tensor`, representing the worker index.
#'
#' @return Dataset: A `Dataset`.
#'
#' @section Raises:
#' InvalidArgumentError: if `num_shards` or `index` are illegal values. Note: error checking is done on a best-effort basis, and errors aren't guaranteed to be caught upon dataset creation. (e.g. providing in a placeholder tensor bypasses the early checking, and will instead result in an error during a session.run call.)
#'
#' @export
shard <- function(num_shards, index) {

  python_function_result <- tf_io$IODataset$shard(
    num_shards = num_shards,
    index = index
  )

}
#' @title shuffle
#'
#' @description Randomly shuffles the elements of this dataset.
#'
#' @details This dataset fills a buffer with `buffer_size` elements, then randomly
#' samples elements from this buffer, replacing the selected elements with new
#' elements. For perfect shuffling, a buffer size greater than or equal to the
#' full size of the dataset is required. For instance, if your dataset contains 10,000 elements but `buffer_size` is
#' set to 1,000, then `shuffle` will initially select a random element from
#' only the first 1,000 elements in the buffer. Once an element is selected,
#' its space in the buffer is replaced by the next (i.e. 1,001-st) element,
#' maintaining the 1,000 element buffer. `reshuffle_each_iteration` controls whether the shuffle order should be
#' different for each epoch. In TF 1.X, the idiomatic way to create epochs
#' was through the `repeat` transformation: >>> dataset = tf.data.Dataset.range(3)
#' >>> dataset = dataset.shuffle(3, reshuffle_each_iteration=TRUE)
#' >>> dataset = dataset.repeat(2) # doctest: +SKIP
#' [1, 0, 2, 1, 2, 0] >>> dataset = tf.data.Dataset.range(3)
#' >>> dataset = dataset.shuffle(3, reshuffle_each_iteration=FALSE)
#' >>> dataset = dataset.repeat(2) # doctest: +SKIP
#' [1, 0, 2, 1, 0, 2] In TF 2.0, `tf.data.Dataset` objects are Python iterables which makes it
#' possible to also create epochs through Python iteration: >>> dataset = tf.data.Dataset.range(3)
#' >>> dataset = dataset.shuffle(3, reshuffle_each_iteration=TRUE)
#' >>> list(dataset.as_numpy_iterator()) # doctest: +SKIP
#' [1, 0, 2]
#' >>> list(dataset.as_numpy_iterator()) # doctest: +SKIP
#' [1, 2, 0] >>> dataset = tf.data.Dataset.range(3)
#' >>> dataset = dataset.shuffle(3, reshuffle_each_iteration=FALSE)
#' >>> list(dataset.as_numpy_iterator()) # doctest: +SKIP
#' [1, 0, 2]
#' >>> list(dataset.as_numpy_iterator()) # doctest: +SKIP
#' [1, 0, 2]
#'
#' @param buffer_size A `tf.int64` scalar `tf.Tensor`, representing the number of elements from this dataset from which the new dataset will sample.
#' @param seed (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random seed that will be used to create the distribution. See `tf.compat.v1.set_random_seed` for behavior.
#' @param reshuffle_each_iteration (Optional.) A boolean, which if true indicates that the dataset should be pseudorandomly reshuffled each time it is iterated over. (Defaults to `TRUE`.)
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
shuffle <- function(buffer_size, seed = NULL, reshuffle_each_iteration = NULL) {

  python_function_result <- tf_io$IODataset$shuffle(
    buffer_size = buffer_size,
    seed = seed,
    reshuffle_each_iteration = reshuffle_each_iteration
  )

}
#' @title skip
#'
#' @description Creates a `Dataset` that skips `count` elements from this dataset.
#'
#' @details >>> dataset = tf.data.Dataset.range(10)
#' >>> dataset = dataset.skip(7)
#' >>> list(dataset.as_numpy_iterator())
#' [7, 8, 9]
#'
#' @param count A `tf.int64` scalar `tf.Tensor`, representing the number of elements of this dataset that should be skipped to form the new dataset. If `count` is greater than the size of this dataset, the new dataset will contain no elements. If `count` is -1, skips the entire dataset.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
skip <- function(count) {

  python_function_result <- tf_io$IODataset$skip(
    count = count
  )

}
#' @title stream
#'
#' @description Obtain a non-repeatable StreamIODataset to be used.
#'
#' @details Returns: A class of `StreamIODataset`.
#'
#' @param cls cls
#'
#' @return A class of `StreamIODataset`.
#'
#' @export
stream <- function(cls) {

  python_function_result <- tf_io$IODataset$stream(
    cls = cls
  )

}
#' @title take
#'
#' @description Creates a `Dataset` with at most `count` elements from this dataset.
#'
#' @details >>> dataset = tf.data.Dataset.range(10)
#' >>> dataset = dataset.take(3)
#' >>> list(dataset.as_numpy_iterator())
#' [0, 1, 2]
#'
#' @param count A `tf.int64` scalar `tf.Tensor`, representing the number of elements of this dataset that should be taken to form the new dataset. If `count` is -1, or if `count` is greater than the size of this dataset, the new dataset will contain all elements of this dataset.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
take <- function(count) {

  python_function_result <- tf_io$IODataset$take(
    count = count
  )

}
#' @title window
#'
#' @description Combines (nests of) input elements into a dataset of (nests of) windows.
#'
#' @details A "window" is a finite dataset of flat elements of size `size` (or possibly
#' fewer if there are not enough input elements to fill the window and
#' `drop_remainder` evaluates to false). The `stride` argument determines the stride of the input elements, and the
#' `shift` argument determines the shift of the window. >>> dataset = tf.data.Dataset.range(7).window(2)
#' >>> for window in dataset:
#' ... print(list(window.as_numpy_iterator()))
#' [0, 1]
#' [2, 3]
#' [4, 5]
#' [6]
#' >>> dataset = tf.data.Dataset.range(7).window(3, 2, 1, TRUE)
#' >>> for window in dataset:
#' ... print(list(window.as_numpy_iterator()))
#' [0, 1, 2]
#' [2, 3, 4]
#' [4, 5, 6]
#' >>> dataset = tf.data.Dataset.range(7).window(3, 1, 2, TRUE)
#' >>> for window in dataset:
#' ... print(list(window.as_numpy_iterator()))
#' [0, 2, 4]
#' [1, 3, 5]
#' [2, 4, 6] Note that when the `window` transformation is applied to a dataset of
#' nested elements, it produces a dataset of nested windows. >>> nested = ([1, 2, 3, 4], [5, 6, 7, 8])
#' >>> dataset = tf.data.Dataset.from_tensor_slices(nested).window(2)
#' >>> for window in dataset:
#' ... def to_numpy(ds):
#' ... return list(ds.as_numpy_iterator())
#' ... print(list(to_numpy(component) for component in window))
#' ([1, 2], [5, 6])
#' ([3, 4], [7, 8]) >>> dataset = tf.data.Dataset.from_tensor_slices({'a': [1, 2, 3, 4]})
#' >>> dataset = dataset.window(2)
#' >>> for window in dataset:
#' ... def to_numpy(ds):
#' ... return list(ds.as_numpy_iterator())
#' ... print({'a': to_numpy(window['a'])})
#' {'a': [1, 2]}
#' {'a': [3, 4]}
#'
#' @param size A `tf.int64` scalar `tf.Tensor`, representing the number of elements of the input dataset to combine into a window.
#' @param shift (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the forward shift of the sliding window in each iteration. Defaults to `size`.
#' @param stride (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the stride of the input elements in the sliding window.
#' @param drop_remainder (Optional.) A `tf.bool` scalar `tf.Tensor`, representing whether a window should be dropped in case its size is smaller than `window_size`.
#'
#' @return Dataset: A `Dataset` of (nests of) windows -- a finite datasets of flat elements created from the (nests of) input elements.
#'
#' @export
window <- function(size, shift = NULL, stride = 1L, drop_remainder = FALSE) {

  python_function_result <- tf_io$IODataset$window(
    size = size,
    shift = shift,
    stride = stride,
    drop_remainder = drop_remainder
  )

}
#' @title with_options
#'
#' @description Returns a new `tf.data.Dataset` with the given options set.
#'
#' @details The options are "global" in the sense they apply to the entire dataset.
#' If options are set multiple times, they are merged as long as different
#' options do not use different non-default values. >>> ds = tf.data.Dataset.range(5)
#' >>> ds = ds.interleave(lambda x: tf.data.Dataset.range(5),
#' ... cycle_length=3,
#' ... num_parallel_calls=3)
#' >>> options = tf.data.Options()
#' >>> # This will make the interleave order non-deterministic.
#' >>> options.experimental_deterministic = FALSE
#' >>> ds = ds.with_options(options)
#'
#' @param options A `tf.data.Options` that identifies the options the use.
#'
#' @return Dataset: A `Dataset` with the given options.
#'
#' @section Raises:
#' ValueError: when an option is set more than once to a non-default value
#'
#' @export
with_options <- function(options) {

  python_function_result <- tf_io$IODataset$with_options(
    options = options
  )

}
#' @title zip
#'
#' @description Creates a `Dataset` by zipping together the given datasets.
#'
#' @details This method has similar semantics to the built-in `zip()` function
#' in Python, with the main difference being that the `datasets`
#' argument can be an arbitrary nested structure of `Dataset` objects. >>> # The nested structure of the `datasets` argument determines the
#' >>> # structure of elements in the resulting dataset.
#' >>> a = tf.data.Dataset.range(1, 4) # ==> [ 1, 2, 3 ]
#' >>> b = tf.data.Dataset.range(4, 7) # ==> [ 4, 5, 6 ]
#' >>> ds = tf.data.Dataset.zip((a, b))
#' >>> list(ds.as_numpy_iterator())
#' [(1, 4), (2, 5), (3, 6)]
#' >>> ds = tf.data.Dataset.zip((b, a))
#' >>> list(ds.as_numpy_iterator())
#' [(4, 1), (5, 2), (6, 3)]
#' >>>
#' >>> # The `datasets` argument may contain an arbitrary number of datasets.
#' >>> c = tf.data.Dataset.range(7, 13).batch(2) # ==> [ [7, 8],
#' ... # [9, 10],
#' ... # [11, 12] ]
#' >>> ds = tf.data.Dataset.zip((a, b, c))
#' >>> for element in ds.as_numpy_iterator():
#' ... print(element)
#' (1, 4, array([7, 8]))
#' (2, 5, array([ 9, 10]))
#' (3, 6, array([11, 12]))
#' >>>
#' >>> # The number of elements in the resulting dataset is the same as
#' >>> # the size of the smallest dataset in `datasets`.
#' >>> d = tf.data.Dataset.range(13, 15) # ==> [ 13, 14 ]
#' >>> ds = tf.data.Dataset.zip((a, d))
#' >>> list(ds.as_numpy_iterator())
#' [(1, 13), (2, 14)]
#'
#' @param datasets A nested structure of datasets.
#'
#' @return Dataset: A `Dataset`.
#'
#' @export
zip <- function(datasets) {

  python_function_result <- tf_io$IODataset$zip(
    datasets = datasets
  )

}
