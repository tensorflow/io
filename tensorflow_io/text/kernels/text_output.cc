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
#include "kernels/output_ops.h"

namespace tensorflow {
namespace data {
namespace {

class TextDatasetOutput {
 public:
  Status Final(WritableFile *file) {
    return file->Flush();
  }
  Status Write(WritableFile *file, std::vector<Tensor>& components) {
    // TODO (yongtang): In case of TextDataset the shapes of output can
    // only be either a scalar (`batch = 0`) or an 1-D tensor (`batch != 0`)
    // so the following could be simplified by just calling flat<string>.
    // Once other dataset has been introduced for write then additional
    // logic would be neededto handle `batch = 0` vs `batch != 0` case.
    for (int64 i = 0; i < components[0].NumElements(); i++) {
      TF_RETURN_IF_ERROR(file->Append(components[0].flat<string>()(i)));
      TF_RETURN_IF_ERROR(file->Append("\n"));
    }
    return Status::OK();
  }

 private:
  BackgroundWorker background_worker_;
};

REGISTER_KERNEL_BUILDER(
    Name("IoTextDatasetOutput").Device(DEVICE_CPU), DatasetOutputOp<TextDatasetOutput>);

}  // namespace
}  // namespace data
}  // namespace tensorflow
