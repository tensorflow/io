# Dicom
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3337331.svg)](https://doi.org/10.5281/zenodo.3337331)

## Getting started
[Example Colab notebook](https://colab.research.google.com/drive/1MdjXN3XkYs_mSyVtdRK7zaCbzkjGub_B)

## Documentation
This package has two operations which wrap `DCMTK` functions. `decode_dicom_image` decodes the pixel data from DICOM files, and `decode_dicom_data` decodes tag information. `tags` contains useful DICOM tags such as `tags.PatientsName`. We borrow the same tag notation from the [`pydicom`](https://pydicom.github.io/) dicom package.

### Getting DICOM Image Data
```python
io.dicom.decode_dicom_image(
    contents,
    color_dim=False,
    on_error='skip',
    scale='preserve'
    dtype=tf.uint16
    name=None
)
```

 - **`contents`**: A Tensor of type string. 0-D. The byte string encoded DICOM file
 - **`color_dim`**: An optional `bool`. Defaults to `False`. If `True`, a third channel will be appended to all images forming a 3-D tensor. A 1024 x 1024 grayscale image will be 1024 x 1024 x 1
 - **`on_error`**: Defaults to `skip`. This attribute establishes the behavior in case an error occurs on opening the image or if the output type cannot accomodate all the possible input values. For example if the user sets the output dtype to `tf.uint8`, but a dicom image stores a `tf.uint16` type. `strict` throws an error. `skip` returns a 1-D empty tensor.  `lossy` continues with the operation scaling the value via the `scale` attribute. 
 - **`scale`**:  Defaults to `preserve`. This attribute establishes what to do with the scale of the input values. `auto` will autoscale the input values, if the output type is integer, `auto` will use the maximum output scale for example a `uint8` which stores values from [0, 255] can be linearly stretched to fill a `uint16` that is [0,65535]. If the output is float, `auto` will scale to [0,1]. `preserve` keeps the values as they are, an input value greater than the maximum possible output will be clipped. 
 - **`dtype`**: An optional `tf.DType` from: `tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.float16, tf.float32, tf.float64`. Defaults to `tf.uint16`. 
 - **`name`**: A name for the operation (optional).
 
 **Returns**

A `Tensor` of type `dtype` and the shape is determined by the DICOM file. 

 ### Getting DICOM Tag Data
 
```python
io.dicom.decode_dicom_data(
    contents,
    tags=None,
    name=None
)
```

 - **`contents`**: A Tensor of type string. 0-D. The byte string encoded DICOM file
 - **`tags`**: A Tensor of type `tf.uint32` of any dimension. These `uint32` numbers map directly to DICOM tags
 - **`name`**: A name for the operation (optional).

**Returns**

A `Tensor` of type `tf.string` and same shape as `tags`.  If a dicom tag is a list of strings, they are combined into one string and seperated by a double backslash `\\`. There is a bug in [DCMTK](https://support.dcmtk.org/docs/) if the tag is a list of numbers, only the zeroth element will be returned as a string.



# Bibtex

If this package helped, please kindly cite the below:

```
@misc{marcelo_lerendegui_2019_3337331,
  author       = {Marcelo Lerendegui and
                  Ouwen Huang},
  title        = {Tensorflow Dicom Decoder},
  month        = jul,
  year         = 2019,
  doi          = {10.5281/zenodo.3337331},
  url          = {https://doi.org/10.5281/zenodo.3337331}
}
```

# License

Copyright 2019 Marcelo Lerendegui, Ouwen Huang, Gradient Health Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
