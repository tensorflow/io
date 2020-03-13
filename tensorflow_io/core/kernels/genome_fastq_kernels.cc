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

#include <string>
#include <vector>

#include <utility>
#include "nucleus/io/fastq_reader.h"
#include "nucleus/platform/types.h"
#include "nucleus/protos/fastq.pb.h"
#include "nucleus/util/utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

namespace {
using nucleus::FastqReader;
using nucleus::genomics::v1::FastqRecord;
}  // namespace

class FastqOp : public OpKernel {
 public:
  explicit FastqOp(OpKernelConstruction* context) : OpKernel(context) {}
  ~FastqOp() {}

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const std::string& filename = filename_tensor.scalar<tstring>()();

    std::unique_ptr<FastqReader> reader =
        std::move(FastqReader::FromFile(
                      filename, nucleus::genomics::v1::FastqReaderOptions())
                      .ValueOrDie());

    std::vector<std::string> sequences;
    std::vector<std::string> quality;

    std::shared_ptr<nucleus::FastqIterable> fastq_iterable =
        reader->Iterate().ValueOrDie();
    for (const nucleus::StatusOr<FastqRecord*> maybe_sequence :
         fastq_iterable) {
      OP_REQUIRES(
          context, maybe_sequence.ok(),
          errors::Internal("internal error: ", maybe_sequence.error_message()));
      sequences.push_back(maybe_sequence.ValueOrDie()->sequence());
      quality.push_back(maybe_sequence.ValueOrDie()->quality());
    }

    TensorShape output_shape({static_cast<int64>(sequences.size())});
    Tensor* output_tensor;
    Tensor* quality_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &quality_tensor));

    for (size_t i = 0; i < sequences.size(); i++) {
      output_tensor->flat<tstring>()(i) = std::move(sequences[i]);
      quality_tensor->flat<tstring>()(i) = std::move(quality[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("IO>ReadFastq").Device(DEVICE_CPU), FastqOp);

}  // namespace tensorflow
