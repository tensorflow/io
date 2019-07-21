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

#include "kernels/dataset_ops.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"

namespace tensorflow {
namespace data {

class FileContentInput: public FileInput<bool> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<bool>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (record_to_read != 1) {
      return errors::InvalidArgument("FileDataset only accept reading one record at a time");
    }
    if (state.get() == nullptr) {
      state.reset(new bool(true));
    } else {
      // We only read file once.
      return Status::OK();
    }
    int64 chunk_size = 4096;
    SizedRandomAccessInputStreamInterface* sized_stream = dynamic_cast<SizedRandomAccessInputStreamInterface*>(s);
    if (sized_stream != nullptr) {
      // First try to find out the size of the file, depending on if size is available, we will set the chunk to read.
      uint64 file_size = 0;
      if (sized_stream->GetFileSize(&file_size) == Status::OK()) {
        chunk_size = file_size;
      }
    }
    std::vector<string> entries;
    Status status = Status::OK();
    int64 total_size = 0;
    while (status.ok()) {
      string buffer;
      status = s->ReadNBytes(chunk_size, &buffer);
      if (status.ok() || errors::IsOutOfRange(status)) {
        total_size += buffer.size();
        entries.emplace_back(std::move(buffer));
      }
    }
    if (!errors::IsOutOfRange(status)) {
      return status;
    }
    Tensor value_tensor(ctx->allocator({}), DT_STRING, {1});
    if (entries.size() == 1) {
      value_tensor.flat<string>()((*record_read)) = std::move(entries[0]);
    } else {
      string buffer;
      buffer.reserve(total_size);
      for (size_t i = 0; i < entries.size(); ++i) {
        buffer.append(entries[i]);
      }
      value_tensor.flat<string>()((*record_read)) = std::move(buffer);
    }
    (*record_read)++;
    out_tensors->emplace_back(std::move(value_tensor));
    return Status::OK();
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

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(FileContentInput, "tensorflow::data::FileContentInput");

REGISTER_KERNEL_BUILDER(Name("FileInput").Device(DEVICE_CPU),
                        FileInputOp<FileContentInput>);
REGISTER_KERNEL_BUILDER(Name("FileDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<FileContentInput, bool>);
}  // namespace data
}  // namespace tensorflow
