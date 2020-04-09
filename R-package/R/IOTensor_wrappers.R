#' @title from_arrow
#'
#' @description Creates an `IOTensor` from a pyarrow.Table.
#'
#' @details 
#'
#' @param cls cls
#' @param table An instance of a `pyarrow.Table`.
#'
#' @return A `IOTensor`.
#'
#' @export
from_arrow <- function(cls, table) {

  python_function_result <- tf_io$IOTensor$from_arrow(
    cls = cls,
    table = table
  )

}
#' @title from_audio
#'
#' @description Creates an `IOTensor` from an audio file.
#'
#' @details The following audio file formats are supported:
#' - WAV
#' - OGG
#'
#' @param cls cls
#' @param filename A string, the filename of an audio file.
#'
#' @return A `IOTensor`.
#'
#' @section The following audio file formats are supported:
#' - WAV - OGG
#'
#' @export
from_audio <- function(cls, filename) {

  python_function_result <- tf_io$IOTensor$from_audio(
    cls = cls,
    filename = filename
  )

}
#' @title from_avro
#'
#' @description Creates an `IOTensor` from an avro file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of an avro file.
#' @param schema A string, the schema of an avro file.
#'
#' @return A `IOTensor`.
#'
#' @export
from_avro <- function(cls, filename, schema) {

  python_function_result <- tf_io$IOTensor$from_avro(
    cls = cls,
    filename = filename,
    schema = schema
  )

}
#' @title from_csv
#'
#' @description Creates an `IOTensor` from an csv file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of an csv file.
#'
#' @return A `IOTensor`.
#'
#' @export
from_csv <- function(cls, filename) {

  python_function_result <- tf_io$IOTensor$from_csv(
    cls = cls,
    filename = filename
  )

}
#' @title from_feather
#'
#' @description Creates an `IOTensor` from an feather file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of an feather file.
#'
#' @return A `IOTensor`.
#'
#' @export
from_feather <- function(cls, filename) {

  python_function_result <- tf_io$IOTensor$from_feather(
    cls = cls,
    filename = filename
  )

}
#' @title from_ffmpeg
#'
#' @description Creates an `IOTensor` from a audio/video file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a audio/video file.
#'
#' @return A `IOTensor`.
#'
#' @export
from_ffmpeg <- function(cls, filename) {

  python_function_result <- tf_io$IOTensor$from_ffmpeg(
    cls = cls,
    filename = filename
  )

}
#' @title from_hdf5
#'
#' @description Creates an `IOTensor` from an hdf5 file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of an hdf5 file.
#' @param spec A dict of `dataset:tf.TensorSpec` or `dataset:dtype` pairs that specify the dataset selected and the tf.TensorSpec or dtype of the dataset. In eager mode the spec is probed automatically. In graph mode spec has to be specified.
#'
#' @return A `IOTensor`.
#'
#' @export
from_hdf5 <- function(cls, filename, spec = NULL) {

  python_function_result <- tf_io$IOTensor$from_hdf5(
    cls = cls,
    filename = filename,
    spec = spec
  )

}
#' @title from_json
#'
#' @description Creates an `IOTensor` from an json file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of an json file.
#'
#' @return A `IOTensor`.
#'
#' @export
from_json <- function(cls, filename) {

  python_function_result <- tf_io$IOTensor$from_json(
    cls = cls,
    filename = filename
  )

}
#' @title from_kafka
#'
#' @description Creates an `IOTensor` from a Kafka stream.
#'
#' @details 
#'
#' @param cls cls
#' @param topic A `tf.string` tensor containing topic subscription.
#' @param partition A `tf.int64` tensor containing the partition, by default 0.
#' @param servers An optional list of bootstrap servers, by default `localhost:9092`.
#' @param configuration An optional `tf.string` tensor containing configurations in [Key=Value] format. There are three types of configurations: Global configuration: please refer to 'Global configuration properties' in librdkafka doc. Examples include ["enable.auto.commit=false", "heartbeat.interval.ms=2000"] Topic configuration: please refer to 'Topic configuration properties' in librdkafka doc. Note all topic configurations should be prefixed with `configuration.topic.`. Examples include ["conf.topic.auto.offset.reset=earliest"]
#'
#' @return A `IOTensor`.
#'
#' @export
from_kafka <- function(cls, topic, partition = 0L, servers = NULL, configuration = NULL) {

  python_function_result <- tf_io$IOTensor$from_kafka(
    cls = cls,
    topic = topic,
    partition = partition,
    servers = servers,
    configuration = configuration
  )

}
#' @title from_lmdb
#'
#' @description Creates an `IOTensor` from a LMDB key/value store.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a LMDB file.
#'
#' @return A `IOTensor`.
#'
#' @export
from_lmdb <- function(cls, filename) {

  python_function_result <- tf_io$IOTensor$from_lmdb(
    cls = cls,
    filename = filename
  )

}
#' @title from_parquet
#'
#' @description Creates an `IOTensor` from a parquet file.
#'
#' @details 
#'
#' @param cls cls
#' @param filename A string, the filename of a parquet file.
#'
#' @return A `IOTensor`.
#'
#' @export
from_parquet <- function(cls, filename) {

  python_function_result <- tf_io$IOTensor$from_parquet(
    cls = cls,
    filename = filename
  )

}
#' @title from_tensor
#'
#' @description Converts a `tf.Tensor` into a `IOTensor`.
#'
#' @details Examples: ```python
#' ```
#'
#' @param cls cls
#' @param tensor The `Tensor` to convert.
#'
#' @return A `IOTensor`.
#'
#' @section Raises:
#' ValueError: If tensor is not a `Tensor`.
#'
#' @export
from_tensor <- function(cls, tensor) {

  python_function_result <- tf_io$IOTensor$from_tensor(
    cls = cls,
    tensor = tensor
  )

}
#' @title from_tiff
#'
#' @description Creates an `IOTensor` from a tiff file.
#'
#' @details Note tiff file may consists of multiple images with different shapes.
#'
#' @param cls cls
#' @param filename A string, the filename of a tiff file.
#'
#' @return A `IOTensor`.
#'
#' @export
from_tiff <- function(cls, filename) {

  python_function_result <- tf_io$IOTensor$from_tiff(
    cls = cls,
    filename = filename
  )

}
#' @title graph
#'
#' @description Obtain a GraphIOTensor to be used in graph mode.
#'
#' @details 
#'
#' @param cls cls
#' @param dtype Data type of the GraphIOTensor.
#'
#' @return A class of `GraphIOTensor`.
#'
#' @export
graph <- function(cls, dtype) {

  python_function_result <- tf_io$IOTensor$graph(
    cls = cls,
    dtype = dtype
  )

}
#' @title spec
#'
#' @description The `TensorSpec` of values in this tensor.
#'
#' @details 
#'


#'
#' @export
 {

  python_function_result <- tf_io$IOTensor$spec(
)

}
