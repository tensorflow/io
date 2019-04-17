# TensorFlow OSS Filesystem Extension

OSS is an object storage service provided by Alibaba Cloud, see [here](https://www.alibabacloud.com/product/oss) for more information about the service.

This module provides an extension that emulates a filesystem using the object storage service. The directory structures are encoded in object keys and file contents are stored in objects. The extension is implemented using [OSS C SDK](https://github.com/aliyun/aliyun-oss-c-sdk).

To use the extension, first save your OSS credential in a file,  in `INI` format:

```
[OSSCredentials]
host = cn-hangzhou.oss.aliyun-inc.com
accessid = your_oss_access_id
accesskey = you_oss_access_key
```

Then set environment variable `OSS_CREDENTIALS` to the path of the file.

In Python code, import the extension `ossfs_op` module to use the extension with `gfile`. The files and directory URI should have `oss://` prefix, followed by a bucket name, then the directory hierarchy.

```
import tensorflow_io.oss.python.ops.ossfs_ops
from tensorflow.python.platform import gfile

gfile.MkDir('oss://your_bucket_name/test_dir')
```

## Test

File `python/tests/ossfs_test.py` contains basic filesystem functionality tests. Besides `OSS_CREDENTIALS`, the tests also require an `OSS_FS_TEST_BUCKET` environment variable containing an accessible bucket name. To build the extension and run tests:

```
bazel build tensorflow_io/oss/...
./bazel-bin/tensorflow_io/oss/ossfs_py_test
```
