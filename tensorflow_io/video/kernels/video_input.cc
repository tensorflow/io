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

#include "kernels/video_reader.h"

namespace tensorflow {
namespace data {

static mutex mu(LINKER_INITIALIZED);
static unsigned count(0);
void VideoReaderInit() {
  mutex_lock lock(mu);
  count++;
  if (count == 1) {
    // Register all formats and codecs
    av_register_all();
  }
}

class VideoInput: public FileInput<video::VideoReader> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<video::VideoReader>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      VideoReaderInit();
      state.reset(new video::VideoReader(dynamic_cast<SizedRandomAccessInputStreamInterface*>(s), filename()));
      TF_RETURN_IF_ERROR(state.get()->ReadHeader());
    }
    // Read the first frame to get height and width
    int num_bytes, height, width;
    uint8_t *value;
    Status status = state.get()->ReadFrame(&num_bytes, &value, &height, &width);
    if (!(status.ok() || errors::IsOutOfRange(status))) {
      return status;
    }
    if (!status.ok()) {
      return Status::OK();
    }
    Tensor value_tensor(ctx->allocator({}), DT_UINT8, {record_to_read, height, width, 3});
    std::memcpy(reinterpret_cast<char*>(value_tensor.flat<uint8_t>().data()), reinterpret_cast<char*>(value), num_bytes * sizeof(uint8_t));
    (*record_read)++;
    while ((*record_read) < record_to_read) {
      Status status = state.get()->ReadFrame(&num_bytes, &value, &height, &width);
      if (!(status.ok() || errors::IsOutOfRange(status))) {
        return status;
      }
      if (!status.ok()) {
        break;
      }
      int64 offset = (*record_read) * height * width * 3;
      std::memcpy(reinterpret_cast<char*>(&value_tensor.flat<uint8_t>().data()[offset]), reinterpret_cast<char*>(value), num_bytes * sizeof(uint8_t));
      (*record_read)++;
    }
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
 protected:
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(VideoInput, "tensorflow::data::VideoInput");

REGISTER_KERNEL_BUILDER(Name("VideoInput").Device(DEVICE_CPU),
                        FileInputOp<VideoInput>);
REGISTER_KERNEL_BUILDER(Name("VideoDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<VideoInput, video::VideoReader>);

}  // namespace data
}  // namespace tensorflow
