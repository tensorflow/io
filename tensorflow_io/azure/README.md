# Azure storage integration

## Configuration via environment variables
- `TF_AZURE_STORAGE_KEY`
- `TF_AZURE_STORAGE_USE_HTTPS`
- `TF_AZURE_STORAGE_BLOB_ENDPOINT`

```python
>>> import os
>>> import tensorflow as tf
>>> import tensorflow_io.azure.python.ops.azfs_ops
>>>
>>> os.environ['TF_AZURE_STORAGE_KEY'] = 'my-storage-key'
>>>
>>> with tf.gfile.Open("az://myaccount/mycontainer/hello.txt", mode='w') as w:
>>>   w.write("Hello, world!")
>>>
>>> with tf.gfile.Open("az://myaccount/mycontainer/hello.txt", mode='r') as r:
>>>   print(r.read())

Hello, world!
```

## TODO

- Support OAuth
