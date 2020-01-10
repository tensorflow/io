# Release 0.11.0

## Major Features
* Add Ogg audio format support.
* Add FLAC audio format support.
* Add HDR image format support.
* Extend HDF5 format support to all common types.

## Thanks to our Contributors

This release contains contributions from many people:

Mark Daoust, Yong Tang, Yuan Tang, pshiko

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.10.0

## Major Features
* Add Kafka message key support.
* Add support to calculate Phred Quality Scores (Genome).
* Add PBM(PPM/PGM) image support.
* Add OpenEXR format support.
* Use balanced sharding strategy in bigquery read session.
* Add `decode_jpeg_exif` to support extract orientation information.
* Enable SSE4.2 and AVX for macOS and Linux build.

## Thanks to our Contributors

This release contains contributions from many people:

Aleksey Vlasenko, Amarpreet Singh, Mark Daoust, Olivier Martin, Peter Götz,
Soroush Radpour, Suyash Kumar, Yong Tang, Yuan Tang

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.9.1

## Major Features
* `tensorflow_io.bigquery`: Fixes for Google Cloud BigQuery for Tensorflow 2.0

## Thanks to our Contributors

This release contains contributions from many people:

Aleksey Vlasenko, Amarpreet Singh, Bryan Cutler, Damien Pontifex, Duke Wang,
Jiacheng Xu, Marcelo Lerendegui, Mark Daoust, Ouwen Huang, Suyash Kumar,
Yong Tang, Yuan Tang, henrytansetiawan

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.9.0

Release 0.9.0 is the first release that is fully compatible with
TensorFlow 2.0. The package of 0.9.0 is different from 0.8.0,
though it shares the same code base as 0.8.0 with the same sets
of features and contributors.

## Major Features
* **TensorFlow 2.0 compatible**.
* `tensorflow_io.json`: JSON Dataset support.
* `tensorflow_io.dicom`: Medical image DICOM format support.
* `tensorflow_io.genome`: DNA Sequence FastQ format support added.
* `tensorflow_io.ffmpeg`: FFmpeg now support selection of substreams.
* `tensorflow_io.ffmpeg`: FFmpeg now support subtitle (captioning).
* `tensorflow_io.ffmpeg`: FFmpeg now support decode video from memory.
* `tensorflow_io.image`: BMP encoding (encode_bmp) support.
* `tensorflow_io.kafka`: Kafka Dataset now support Kafka Schema Registry.
* `tensorflow_io.audio`: WAV format now support 24 bit audio streams.
* `tensorflow_io.text`: Regex capture group (`re2_full_match`) support.
* manylinux2010 compliant on Linux.

## Thanks to our Contributors

This release contains contributions from many people:

Aleksey Vlasenko, Amarpreet Singh, Bryan Cutler, Damien Pontifex, Duke Wang,
Jiacheng Xu, Marcelo Lerendegui, Mark Daoust, Ouwen Huang, Suyash Kumar,
Yong Tang, Yuan Tang, henrytansetiawan

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.8.1

## Major Features
* `tensorflow_io.bigquery`: Fixes for Google Cloud BigQuery for Tensorflow 1.15.

## Thanks to our Contributors

This release contains contributions from many people:

Aleksey Vlasenko, Amarpreet Singh, Bryan Cutler, Damien Pontifex, Duke Wang,
Jiacheng Xu, Marcelo Lerendegui, Mark Daoust, Ouwen Huang, Suyash Kumar,
Yong Tang, Yuan Tang, henrytansetiawan

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.8.0

## Major Features
* `tensorflow_io.json`: JSON Dataset support.
* `tensorflow_io.dicom`: Medical image DICOM format support.
* `tensorflow_io.genome`: DNA Sequence FastQ format support added.
* `tensorflow_io.ffmpeg`: FFmpeg now support selection of substreams.
* `tensorflow_io.ffmpeg`: FFmpeg now support subtitle (captioning).
* `tensorflow_io.ffmpeg`: FFmpeg now support decode video from memory.
* `tensorflow_io.image`: BMP encoding (encode_bmp) support.
* `tensorflow_io.kafka`: Kafka Dataset now support Kafka Schema Registry.
* `tensorflow_io.audio`: WAV format now support 24 bit audio streams.
* `tensorflow_io.text`: Regex capture group (`re2_full_match`) support.
* manylinux2010 compliant on Linux.

## Thanks to our Contributors

This release contains contributions from many people:

Aleksey Vlasenko, Amarpreet Singh, Bryan Cutler, Damien Pontifex, Duke Wang,
Jiacheng Xu, Marcelo Lerendegui, Mark Daoust, Ouwen Huang, Suyash Kumar,
Yong Tang, Yuan Tang, henrytansetiawan

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.7.2

## Major Features
* `tensorflow_io.bigquery`: Fixes for Google Cloud BigQuery for Tensorflow 1.14

## Thanks to our Contributors

This release contains contributions from many people:

Aleksey Vlasenko, Anton Dmitriev, Bryan Cutler, Damien Pontifex, Ivelin Ivanov,
Jiacheng Xu, Misha Brukman, Russell Power, Yong Tang, Yuan Tang, zhjunqin

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.7.1

## Major Features
* `tensorflow_io.bigquery`: Fixes for Google Cloud BigQuery for Tensorflow 1.14

## Thanks to our Contributors

This release contains contributions from many people:

Aleksey Vlasenko, Anton Dmitriev, Bryan Cutler, Damien Pontifex, Ivelin Ivanov,
Jiacheng Xu, Misha Brukman, Russell Power, Yong Tang, Yuan Tang, zhjunqin

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.7.0

## Major Features
* `tensorflow_io.bigquery`: Google Cloud BigQuery support.
* `tensorflow_io.text`: Pcap network packet capture file support.
* `tensorflow_io.azure`: Microsoft Azure Storage support.
* `tensorflow_io.gcs`: GCS Configuration support.
* `tensorflow_io.prometheus`: Prometheus observation data support.

## Thanks to our Contributors

This release contains contributions from many people:

Aleksey Vlasenko, Anton Dmitriev, Bryan Cutler, Damien Pontifex, Ivelin Ivanov,
Jiacheng Xu, Misha Brukman, Russell Power, Yong Tang, Yuan Tang, zhjunqin

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.6.0

## Major Features
* `tensorflow_io.avro`: Apache Avro Dataset.
* `tensorflow_io.audio`: WAV file Dataset.
* `tensorflow_io.grpc`: gRPC server Dataset, support for streaming Numpy input.
* `tensorflow_io.hdf5`: HDF5 file Dataset.
* `tensorflow_io.text`: Text file Dataset and TextSequence output.
* Improved batching support for many Datasets, see [#191](https://github.com/tensorflow/io/issues/191).

## Thanks to our Contributors

This release contains contributions from many people:

Yong Tang, Yuan (Terry) Tang, Bryan Cutler, Jiacheng Xu, Junqin Zhang,
August Xiong, caszkgui, zou000

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.5.0

## Major Features
* `tensorflow_io.kafka`: Kafka Output support.
* `tensorflow_io.cifar`: CIFAR file format support.
* `tensorflow_io.bigtable`: Google Cloud Bigtable support.

## Thanks to our Contributors

This release contains contributions from many people:

Bryan Cutler, Damien Pontifex, Florian Raudies, Henry Tan,
Junqin Zhang, Stephan Uphoff, Yong Tang, Yuan (Terry) Tang

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.4.0

## Major Features
* `tensorflow_io.pubsub`: Google Cloud Pub/Sub Dataset support.
* `tensorflow_io.mnist`: MNIST file format support.
* `tensorflow_io.image`: `decode_webp` support.
* `tensorflow_io.kafka`: `write_kafka` support.

## Thanks to our Contributors

This release contains contributions from many people:

Bryan Cutler, Jongwook Choi, Sergii Khomenko, Stephan Uphoff,
Yong Tang, Yuan (Terry) Tang

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.3.0

## Major Features
* `tensorflow_io.image`: TIFF image Dataset.
* `tensorflow_io.lmdb`: LMDB Dataset.

## Thanks to our Contributors

This release contains contributions from many people:

Bryan Cutler, Yong Tang, Yuan (Terry) Tang

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.2.0

## Major Features
* `tensorflow_io.arrow`: Apache Arrow Datasets.
* `tensorflow_io.image`: WebP image Dataset.
* `tensorflow_io.libsvm`: LIBSVM Dataset.
* `tensorflow_io.parquet`: Apache Parquet Dataset.
* `tensorflow_io.video`: Video Dataset (from FFmpeg).

## Thanks to our Contributors

This release contains contributions from many people:

Anton Dmitriev, Bryan Cutler, Peng Yu, Yong Tang, Yuan (Terry) Tang

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.1.0

Initial release of TensorFlow I/O.

## Major Features
* `tensorflow_io.ignite`: Data source for Apache Ignite and File System (IGFS).
* `tensorflow_io.kafka`: Apache Kafka stream-processing support.
* `tensorflow_io.kinesis`: Amazon Kinesis data streams support.
* `tensorflow_io.hadoop`: Hadoop SequenceFile format support.

## Thanks to our Contributors

This release contains contributions from many people:

Anjali Sridhar, Anton Dmitriev, Artem Malykh, Brennan Saeta, Derek Murray,
Gunhan Gulsoy, Jacques Pienaar, Jianwei Xie, Jiri Simsa, knight, Loo Rong Jie,
Martin Wicke, Michael Case, Sergei Lebedev, Sourabh Bajaj, Yifei Feng,
Yong Tang, Yuan (Terry) Tang, Yun Peng

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.
