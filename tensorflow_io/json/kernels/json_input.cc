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

#include "kernels/dataset_ops.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include <jsoncpp/json/json.h>

namespace tensorflow {
namespace data {

class JSONInputStream{
public: 
  explicit JSONInputStream(io::InputStreamInterface* s, const std::vector<string>& columns)
    : columns_(columns)
    , input_stream_(nullptr)
    , buffered_stream_(nullptr)
    , file_(nullptr) {
    input_stream_ = dynamic_cast<SizedRandomAccessInputStreamInterface*>(s);
    if (input_stream_ == nullptr) {
      buffered_stream_.reset(new SizedRandomAccessBufferedStream(s));
      input_stream_ = buffered_stream_.get();
    }
  }

  Status ReadRecord(IteratorContext* ctx, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) {
    ##TODO: Implement the mechanism to parse JSON file.
    return Status::OK();
  }
};


class JSONInput: public FileInput<JSONInputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<JSONInputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new JSONInputStream(s, columns()));
      TF_RETURN_IF_ERROR(state.get()->Open());
    }
    return state.get()->ReadRecord(ctx, record_to_read, record_read, out_tensors);
  }
  Status FromStream(io::InputStreamInterface* s) override {
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    return true;
  }
};

}  // namespace data
}  // namespace tensorflow
