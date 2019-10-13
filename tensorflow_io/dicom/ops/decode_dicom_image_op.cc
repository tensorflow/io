/* Copyright 2019 Gradient Health Inc. All Rights Reserved.
   Author: Marcelo Lerendegui <marcelo@gradienthealth.io>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("IO>DecodeDICOMImage")
    .Input("contents: string")
    .Output("output: dtype")
    .Attr(
        "dtype: {uint8, uint16, uint32, uint64, float16, float, double} = "
        "DT_UINT16")
    .Attr("color_dim: bool = true")
    .Attr("on_error: {'strict', 'skip', 'lossy'} = 'skip'")
    .Attr("scale: {'auto', 'preserve'} = 'preserve'")
    .Doc(R"doc(
loads a dicom image file and returns its pixel information in the specified output format
)doc");

}  // namespace tensorflow
