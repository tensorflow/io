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

#include "kernels/audio_ffmpeg_reader.h"

namespace tensorflow {
namespace data {

class AudioInput: public FileInput<audio::AudioReader> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<audio::AudioReader>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      FFmpegReaderInit();
      state.reset(new audio::AudioReader(dynamic_cast<SizedRandomAccessInputStreamInterface*>(s), filename()));
      TF_RETURN_IF_ERROR(state.get()->ReadHeader());
    }
    int64 channels = state.get()->Channels();
    Tensor value_tensor(ctx->allocator({}), DT_INT16, {record_to_read, channels});
    while ((*record_read) < record_to_read) {
      int64 offset = (*record_read) * channels; // Note: int16 based offset, not char
      
      Status status = state.get()->ReadSample(&value_tensor.flat<int16>().data()[offset]);
      if (!(status.ok() || errors::IsOutOfRange(status))) {
        return status;
      }
      if (!status.ok()) {
        break;
      }
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

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(AudioInput, "tensorflow::data::AudioInput");

REGISTER_KERNEL_BUILDER(Name("IO>AudioInput").Device(DEVICE_CPU),
                        FileInputOp<AudioInput>);
REGISTER_KERNEL_BUILDER(Name("IO>AudioDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<AudioInput, audio::AudioReader>);

}  // namespace data
}  // namespace tensorflow
