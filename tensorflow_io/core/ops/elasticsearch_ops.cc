/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("IO>ElasticsearchReadableInit")
    .Input("healthcheck_url: string")
    .Input("healthcheck_field: string")
    .Input("request_url: string")
    .Input("headers: string")
    .Output("resource: resource")
    .Output("columns: string")
    .Output("dtypes: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>ElasticsearchReadableNext")
    .Input("resource: resource")
    .Input("request_url: string")
    .Input("scroll_request_url: string")
    .Output("items: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

}  // namespace
}  // namespace io
}  // namespace tensorflow
