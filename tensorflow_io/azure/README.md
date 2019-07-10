# Azure blob storage integration

## Configuration via environment variables
- `TF_AZURE_USE_DEV_STORAGE`  
   Set to 1 to use local development storage emulator for connections like 'az://devstoreaccount1/container/file.txt'. This will take precendence over all other settings so `unset` to use any other connection
- `TF_AZURE_STORAGE_KEY`  
   Account key for the storage account in use
- `TF_AZURE_STORAGE_USE_HTTP`  
  Set to any value if you don't want to use https transfer. `unset` to use default of https 
- `TF_AZURE_STORAGE_BLOB_ENDPOINT`  
  Set to the endpoint of blob storage - default is .core.windows.net

```python
>>> import os
>>> import tensorflow as tf
>>> import tensorflow_io.azure
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
