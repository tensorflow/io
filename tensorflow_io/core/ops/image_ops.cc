/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace io {
namespace {

REGISTER_OP("IO>DecodeTiffInfo")
    .Input("input: string")
    .Output("shape: int64")
    .Output("dtype: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>DecodeTiff")
    .Input("input: string")
    .Input("index: int64")
    .Output("image: uint8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(
          0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>EncodeBmp")
    .Input("input: uint8")
    .Output("output: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>DecodeWebP")
    .Input("contents: string")
    .Output("image: uint8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(
          0, c->MakeShape({shape_inference::InferenceContext::kUnknownDim,
                           shape_inference::InferenceContext::kUnknownDim, 4}));
      return Status::OK();
    });

REGISTER_OP("IO>DecodePnm")
    .Input("input: string")
    .Output("image: dtype")
    .Attr("dtype: {uint8, uint16} = DT_UINT8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(
          0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>DrawBoundingBoxesV3")
    .Input("images: T")
    .Input("boxes: float")
    .Input("colors: float")
    .Input("texts: string")
    .Attr("font_size: int = 0")
    .Output("output: T")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

REGISTER_OP("IO>DecodeJpegExif")
    .Input("input: string")
    .Output("orientation: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>DecodeExrInfo")
    .Input("input: string")
    .Output("shape: int64")
    .Output("dtype: int64")
    .Output("channel: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      c->set_output(2, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>DecodeExr")
    .Input("input: string")
    .Input("index: int64")
    .Input("channel: string")
    .Output("image: dtype")
    .Attr("dtype: {uint32, half, float}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(
          0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>DecodeHdr")
    .Input("input: string")
    .Output("image: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), 3}));
      return Status::OK();
    });

REGISTER_OP("IO>DecodeDICOMImage")
    .Input("contents: string")
    .Output("output: dtype")
    .Attr(
        "dtype: {uint8, uint16, uint32, uint64, float16, float, double} = "
        "DT_UINT16")
    .Attr("color_dim: bool = true")
    .Attr("on_error: {'strict', 'skip', 'lossy'} = 'skip'")
    .Attr("scale: {'auto', 'preserve'} = 'preserve'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim(),
                                     c->UnknownDim(), c->UnknownDim()}));
      return Status::OK();
    })
    .Doc(R"doc(
loads a dicom image file and returns its pixel information in the specified output format
)doc");

REGISTER_OP("IO>DecodeDICOMData")
    .Input("contents: string")
    .Input("tags: dtype")
    .Attr("dtype: {uint32,string}")
    .Output("tag_values: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
loads a dicom file and returns the specified tags values as string.
)doc");

REGISTER_OP("IO>DecodeNV12")
    .Input("input: string")
    .Input("size: int32")
    .Output("image: uint8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), 3}));
      return Status::OK();
    });

REGISTER_OP("IO>DecodeYUY2")
    .Input("input: string")
    .Input("size: int32")
    .Output("image: uint8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), 3}));
      return Status::OK();
    });

REGISTER_OP("IO>DecodeAVIF")
    .Input("contents: string")
    .Output("image: uint8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(
          0, c->MakeShape({shape_inference::InferenceContext::kUnknownDim,
                           shape_inference::InferenceContext::kUnknownDim, 4}));
      return Status::OK();
    });

}  // namespace
}  // namespace io
}  // namespace tensorflow
