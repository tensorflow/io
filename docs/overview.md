<div align="center">
  <img src="https://tensorflow.org/images/SIGIO.png" width="60%"><br><br>
</div>

-----------------

# TensorFlow I/O

TensorFlow I/O is an extension package to Tensorflow, which encompasses io support for 
a collection of file systems and file formats that are not available in TensorFlow's built-in support.
Integrations with many systems and cloud vendors include (but not limited to):

- Prometheus
- Apache Kafka
- Apache Ignite
- Google Cloud BigQuery
- Google Cloud PubSub
- AWS Kinesis
- Microsoft Azure Storage
- Alibaba Cloud OSS etc.

## S3 and S3-compatible object storage

The `s3://` file system, registered on `import tensorflow_io`, supports Amazon S3
and S3-compatible object stores such as Backblaze B2, Cloudflare R2, and MinIO. It
reads the standard AWS credential and region variables (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`); for a non-AWS provider, set `S3_ENDPOINT`
to the provider endpoint. The same `s3://` paths are used for reading and writing
through `tf.data`, `tf.io.gfile`, `tf.train.Checkpoint`, and `tf.saved_model`.

```python
import tensorflow as tf
import tensorflow_io as tfio  # registers the s3:// file system

dataset = tf.data.TFRecordDataset("s3://my-bucket/train/shard-00000.tfrecord")
```

```bash
export AWS_ACCESS_KEY_ID=<access_key_id>
export AWS_SECRET_ACCESS_KEY=<secret_access_key>
export AWS_REGION=us-west-004
export S3_ENDPOINT=https://s3.us-west-004.backblazeb2.com
```

## Community

* SIG IO [Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/io) and mailing list: [io@tensorflow.org](io@tensorflow.org)
* SIG IO [Monthly Meeting Notes](https://docs.google.com/document/d/1CB51yJxns5WA4Ylv89D-a5qReiGTC0GYum6DU-9nKGo/edit)
* Gitter room: [tensorflow/sig-io](https://gitter.im/tensorflow/sig-io)

## More Information

* [TensorFlow with Apache Arrow Datasets](https://medium.com/tensorflow/tensorflow-with-apache-arrow-datasets-cdbcfe80a59f) - [Bryan Cutler](https://github.com/BryanCutler)
* [How to build a custom Dataset for Tensorflow](https://towardsdatascience.com/how-to-build-a-custom-dataset-for-tensorflow-1fe3967544d8) - [Ivelin Ivanov](https://github.com/ivelin)
* [TensorFlow on Apache Ignite](https://medium.com/tensorflow/tensorflow-on-apache-ignite-99f1fc60efeb) - [Anton Dmitriev](https://github.com/dmitrievanthony)

## License

[Apache License 2.0](https://github.com/tensorflow/io/blob/master/LICENSE)
