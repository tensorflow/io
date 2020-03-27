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

REGISTER_OP("IO>AudioReadableInit")
    .Input("input: string")
    .Output("resource: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>AudioReadableSpec")
    .Input("input: resource")
    .Output("shape: int64")
    .Output("dtype: int64")
    .Output("rate: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({2}));
      c->set_output(1, c->MakeShape({}));
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>AudioReadableRead")
    .Input("input: resource")
    .Input("start: int64")
    .Input("stop: int64")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>AudioResample")
    .Input("input: T")
    .Input("rate_in: int64")
    .Input("rate_out: int64")
    .Output("output: T")
    .Attr("quality: int")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>AudioDecodeWAV")
    .Input("input: string")
    .Input("shape: int64")
    .Output("value: dtype")
    .Attr("dtype: {uint8, int16, int32, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &shape));
      if (!c->RankKnown(shape)) {
        c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
        return Status::OK();
      }
      if (c->Rank(shape) != 2) {
        return errors::InvalidArgument("rank must be two, received ",
                                       c->DebugString(shape));
      }
      c->set_output(0, shape);
      return Status::OK();
    });

REGISTER_OP("IO>AudioEncodeWAV")
    .Input("input: dtype")
    .Input("rate: int64")
    .Output("value: string")
    .Attr("dtype: {uint8, int16, int32, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>AudioDecodeFlac")
    .Input("input: string")
    .Input("shape: int64")
    .Output("value: dtype")
    .Attr("dtype: {uint8, int16, int32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &shape));
      if (!c->RankKnown(shape)) {
        c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
        return Status::OK();
      }
      if (c->Rank(shape) != 2) {
        return errors::InvalidArgument("rank must be two, received ",
                                       c->DebugString(shape));
      }
      c->set_output(0, shape);
      return Status::OK();
    });

REGISTER_OP("IO>AudioEncodeFlac")
    .Input("input: dtype")
    .Input("rate: int64")
    .Output("value: string")
    .Attr("dtype: {uint8, int16, int32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>AudioDecodeVorbis")
    .Input("input: string")
    .Input("shape: int64")
    .Output("value: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &shape));
      if (!c->RankKnown(shape)) {
        c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
        return Status::OK();
      }
      if (c->Rank(shape) != 2) {
        return errors::InvalidArgument("rank must be two, received ",
                                       c->DebugString(shape));
      }
      c->set_output(0, shape);
      return Status::OK();
    });

REGISTER_OP("IO>AudioEncodeVorbis")
    .Input("input: float32")
    .Input("rate: int64")
    .Output("value: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>AudioDecodeMP3")
    .Input("input: string")
    .Input("shape: int64")
    .Output("value: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &shape));
      if (!c->RankKnown(shape)) {
        c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
        return Status::OK();
      }
      if (c->Rank(shape) != 2) {
        return errors::InvalidArgument("rank must be two, received ",
                                       c->DebugString(shape));
      }
      c->set_output(0, shape);
      return Status::OK();
    });

REGISTER_OP("IO>AudioEncodeMP3")
    .Input("input: float32")
    .Input("rate: int64")
    .Output("value: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>AudioDecodeAAC")
    .Input("input: string")
    .Input("shape: int64")
    .Output("value: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &shape));
      if (!c->RankKnown(shape)) {
        c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
        return Status::OK();
      }
      if (c->Rank(shape) != 2) {
        return errors::InvalidArgument("rank must be two, received ",
                                       c->DebugString(shape));
      }
      c->set_output(0, shape);
      return Status::OK();
    });

}  // namespace
}  // namespace io
}  // namespace tensorflow
