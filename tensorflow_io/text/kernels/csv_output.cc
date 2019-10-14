/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "kernels/output_ops.h"
#include "absl/strings/escaping.h"

namespace tensorflow {
namespace data {
namespace {

class CsvDatasetOutput {
 public:
  Status Final(WritableFile *file) {
    return file->Flush();
  }
  Status Write(WritableFile *file, std::vector<Tensor>& components) {
    // Check each components has the same number of elements
    for (size_t i = 1; i < components.size(); i++) {
      if (components[i].NumElements() != components[0].NumElements()) {
        return errors::InvalidArgument("field ", i, " has differnet number of elements ", components[i].NumElements(), " vs. ", components[0].NumElements());
      }
    }
    for (int64 line = 0; line < components[0].NumElements(); line++) {
      for (size_t i = 0; i < components.size(); i++) {
        switch (components[i].dtype()) {
        case DT_INT32:
          TF_RETURN_IF_ERROR(file->Append(strings::StrCat(components[i].flat<int32>()(line))));
          break;
        case DT_INT64:
          TF_RETURN_IF_ERROR(file->Append(strings::StrCat(components[i].flat<int64>()(line))));
          break;
        case DT_FLOAT:
          TF_RETURN_IF_ERROR(file->Append(strings::StrCat(components[i].flat<float>()(line))));
          break;
        case DT_DOUBLE:
          TF_RETURN_IF_ERROR(file->Append(strings::StrCat(components[i].flat<double>()(line))));
          break;
        case DT_STRING:
          TF_RETURN_IF_ERROR(file->Append(absl::CEscape(components[i].flat<string>()(line))));
          break;
        default:
          return errors::InvalidArgument("field ", i, " has unsupported data type: ", components[i].dtype());
        }
        if (i != components.size() - 1) {
          TF_RETURN_IF_ERROR(file->Append(","));
        }
      }
      TF_RETURN_IF_ERROR(file->Append("\n"));
    }
    return Status::OK();
  }

 private:
  BackgroundWorker background_worker_;
};

REGISTER_KERNEL_BUILDER(
    Name("IO>CsvDatasetOutput").Device(DEVICE_CPU), DatasetOutputOp<CsvDatasetOutput>);

}  // namespace
}  // namespace data
}  // namespace tensorflow
