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

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace tensorflow {
namespace data {
namespace {

class DecodeJSONOp : public OpKernel {
 public:
  explicit DecodeJSONOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
  }

  void Compute(OpKernelContext* context) override {
    // TODO: support batch (1-D) input
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<string>()();

    const Tensor* names_tensor;
    OP_REQUIRES_OK(context, context->input("names", &names_tensor));

    OP_REQUIRES(context, (names_tensor->NumElements() == shapes_.size()),
                errors::InvalidArgument(
                    "shapes and names should have same number: ",
                    shapes_.size(), " vs. ", names_tensor->NumElements()));
    rapidjson::Document d;
    d.Parse(input.c_str());
    OP_REQUIRES(context, d.IsObject(),
                errors::InvalidArgument("not a valid JSON object"));
    for (size_t i = 0; i < shapes_.size(); i++) {
      Tensor* value_tensor;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, shapes_[i], &value_tensor));
      rapidjson::Value* entry =
          rapidjson::Pointer(names_tensor->flat<string>()(i).c_str()).Get(d);
      OP_REQUIRES(context, (entry != nullptr),
                  errors::InvalidArgument("no value for ",
                                          names_tensor->flat<string>()(i)));
      if (entry->IsArray()) {
        OP_REQUIRES(context, entry->Size() == value_tensor->NumElements(),
                    errors::InvalidArgument(
                        "number of elements in JSON does not match spec: ",
                        entry->Size(), " vs. ", value_tensor->NumElements()));

        switch (value_tensor->dtype()) {
          case DT_INT32:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<int32>()(j) = (*entry)[j].GetInt();
            }
            break;
          case DT_INT64:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<int64>()(j) = (*entry)[j].GetInt64();
            }
            break;
          case DT_FLOAT:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<float>()(j) = (*entry)[j].GetDouble();
            }
            break;
          case DT_DOUBLE:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<double>()(j) = (*entry)[j].GetDouble();
            }
            break;
          case DT_STRING:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<string>()(j) = (*entry)[j].GetString();
            }
            break;
          default:
            OP_REQUIRES(
                context, false,
                errors::InvalidArgument("data type not supported: ",
                                        DataTypeString(value_tensor->dtype())));
            break;
        }

      } else {
        switch (value_tensor->dtype()) {
          case DT_INT32:
            value_tensor->scalar<int32>()() = entry->GetInt();
            break;
          case DT_INT64:
            value_tensor->scalar<int64>()() = entry->GetInt64();
            break;
          case DT_FLOAT:
            value_tensor->scalar<float>()() = entry->GetDouble();
            break;
          case DT_DOUBLE:
            value_tensor->scalar<double>()() = entry->GetDouble();
            break;
          case DT_STRING:
            value_tensor->scalar<string>()() = entry->GetString();
            break;
          default:
            OP_REQUIRES(
                context, false,
                errors::InvalidArgument("data type not supported: ",
                                        DataTypeString(value_tensor->dtype())));
            break;
        }
      }
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::vector<TensorShape> shapes_ GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodeJSON").Device(DEVICE_CPU), DecodeJSONOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
