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

REGISTER_OP("IO>KafkaReadableInit")
    .Input("topic: string")
    .Input("partition: int32")
    .Input("offset: int64")
    .Input("metadata: string")
    .Output("resource: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>KafkaReadableNext")
    .Input("input: resource")
    .Input("index: int64")
    .Output("message: string")
    .Output("key: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>KafkaReadableRead")
    .Input("input: resource")
    .Input("start: int64")
    .Input("stop: int64")
    .Output("message: string")
    .Output("key: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>KafkaReadableSpec")
    .Input("input: resource")
    .Input("start: int64")
    .Input("stop: int64")
    .Output("start_offset: int64")
    .Output("stop_offset: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>KafkaIterableInit")
    .Input("topic: string")
    .Input("partition: int32")
    .Input("offset: int64")
    .Input("metadata: string")
    .Output("resource: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>LayerKafkaCall")
    .Input("input: T")
    .Input("content: string")
    .Input("resource: resource")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("IO>LayerKafkaInit")
    .Input("topic: string")
    .Input("partition: int32")
    .Input("metadata: string")
    .Output("resource: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>LayerKafkaSync")
    .Input("resource: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>KafkaGroupReadableInit")
    .Input("topics: string")
    .Input("metadata: string")
    .Output("resource: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>KafkaGroupReadableNext")
    .Input("input: resource")
    .Input("index: int64")
    .Input("message_timeout: int64")
    .Input("stream_timeout: int64")
    .Output("message: string")
    .Output("key: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

}  // namespace
}  // namespace io
}  // namespace tensorflow
