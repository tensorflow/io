# TensorFlow OSS Filesystem Extension

OSS is an object storage service provided by Alibaba Cloud, see [here](https://www.alibabacloud.com/product/oss) for more information about the service.

This module provides an extension that emulates a filesystem using the object storage service. The directory structures are encoded in object keys and file contents are stored in objects. The extension is implemented using [OSS C SDK](https://github.com/aliyun/aliyun-oss-c-sdk).

To use the extension, first get your OSS credential from [Alibaba OSS Service](https://www.alibabacloud.com/product/oss), including:

```
host = your_oss_cluster_host
access_id = your_oss_access_id
access_key = you_oss_access_key
bucket = your_oss_bucket_name
```

In Python code, import the extension `ossfs_op` module to use the extension with `gfile`. The files and directory URI should have `oss://` prefix, followed by an oss bucket name, access_id, access_key, oss_host, then the directory hierarchy.

```python
import tensorflow_io.oss
from tensorflow.python.platform import gfile

gfile.MkDir('oss://${bucket}\x01id=${access_id}\x02key=${access_key}\x02host=${host}/test_dir')
```

With the extension installed, OSS files can be used with Dataset Ops, etc., in the same fashion as other files.

```python
dataset = tf.data.TextLineDataset(["oss://${bucket}\x01id=${access_id}\x02key=${access_key}\x02host=${host}/data_dir/file1"])
```

## Test

File `tests/test_oss.py` contains basic filesystem functionality tests. See `README.md` in the root directory for more information about running tests. Make sure OSS credential has been set before running `pytest tests`. You can also just run the OSS test using `pytest tests/test_oss.py`
