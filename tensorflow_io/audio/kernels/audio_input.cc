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

namespace tensorflow {
namespace data {

class WAVInputStream : public io::BufferedInputStream {
public:
  explicit WAVInputStream(InputStreamInterface* input_stream)
    : io::BufferedInputStream(input_stream, 256 * 1024) {
  }
  Status ReadRecord(int64 samples_to_read, int64* samples_read, string* value) {
    while (data_start_ + data_size_ == data_offset_) {
      if (Tell() == file_size_ + 8) {
	*samples_read = 0;
	return Status::OK();
      }
      TF_RETURN_IF_ERROR(ReadNBytes(sizeof(struct DataHeader), &buffer_));
      struct DataHeader *p = (struct DataHeader *)buffer_.data();
      if (memcmp(p->mark, "data", 4) == 0 && p->size != 0) {
	data_size_ = p->size;
	data_start_ = Tell();
	data_offset_ = Tell();
        TF_RETURN_IF_ERROR(ReadNBytes(data_size_, &buffer_));
        continue;
      }
      TF_RETURN_IF_ERROR(SkipNBytes(p->size));
    }
    if (samples_to_read <= 0 || (data_offset_ + samples_to_read * num_channels_ * sizeof(int16) >= data_start_ + data_size_)) {
      samples_to_read = (data_start_ + data_size_ - data_offset_) / num_channels_ / sizeof(int16);
    }
    *value = buffer_.substr(data_offset_ - data_start_, samples_to_read * num_channels_ * sizeof(int16));
    data_offset_ += samples_to_read * num_channels_ * sizeof(int16);
    *samples_read = samples_to_read;
    return Status::OK();
  }
  Status ReadHeader() {
    string buffer;
    TF_RETURN_IF_ERROR(ReadNBytes(sizeof(struct WAVHeader), &buffer));
    struct WAVHeader *header = (struct WAVHeader *)buffer.data();
    if (memcmp(header->riff, "RIFF", 4) != 0) {
      return errors::InvalidArgument("WAV file must starts with `RIFF`");
    }
    file_size_ = header->size;
    if (memcmp(header->wave, "WAVE", 4) != 0) {
      return errors::InvalidArgument("WAV file must contains riff type `WAVE`");
    }
    if (memcmp(header->fmt, "fmt ", 4) != 0) {
      return errors::InvalidArgument("WAV file must contains `fmt ` mark");
    }
    int32 fmt_size_ = header->fmt_size;
    if (fmt_size_ != 16 && fmt_size_ != 18) {
      return errors::InvalidArgument("WAV file must have `fmt_size ` 16 or 18, received", fmt_size_);
    }
    int16 fmt_type_ = header->fmt_type;
    if (fmt_type_ != 1) {
      return errors::InvalidArgument("WAV file must have `fmt_type ` 1, received", fmt_type_);
    }
    num_channels_ = header->num_channels;
    if (num_channels_ <= 0) {
      return errors::InvalidArgument("WAV file have invalide channels: ", num_channels_);
    }
    int32 sample_rate_ = header->sample_rate;
    int32 byte_rate_ = header->byte_rate;
    int16 sample_alignment_ = header->sample_alignment;
    int16 bit_depth_ = header->bit_depth;
    if (bit_depth_ != 16) {
      return errors::InvalidArgument("WAV file must contains 16 bits data");
    }
    if (fmt_size_ == 18) {
      TF_RETURN_IF_ERROR(SkipNBytes(2));
    }
    do {
      TF_RETURN_IF_ERROR(ReadNBytes(sizeof(struct DataHeader), &buffer));
      struct DataHeader *p = (struct DataHeader *)buffer.data();
      if (memcmp(p->mark, "data", 4) == 0) {
	data_size_ = p->size;
	data_start_ = Tell();
	data_offset_ = Tell();
        TF_RETURN_IF_ERROR(ReadNBytes(data_size_, &buffer_));
        return Status::OK();
      }
      TF_RETURN_IF_ERROR(SkipNBytes(p->size));
    } while (Tell() < file_size_ + 8);

    return Status::OK();
  }
  int64 Channel() {
    return num_channels_;
  }
private:
  struct WAVHeader {
    char riff[4]; // "RIFF"
    int32 size; // Size after (file size - 8)
    char wave[4]; // "WAVE"
    char fmt[4]; // "fmt "
    int32 fmt_size; // 16 for PCM
    int16 fmt_type; // 1 for PCM. 3 for IEEE Float
    int16 num_channels;
    int32 sample_rate;
    int32 byte_rate; // Number of bytes per second.
    int16 sample_alignment; // num_channels * Bytes Per Sample
    int16 bit_depth; // Number of bits per sample
  };
  struct DataHeader {
    char mark[4];
    int32 size;
  };
  int64 num_channels_;
  int64 file_size_;
  int64 data_size_;
  int64 data_start_;
  int64 data_offset_;
  string buffer_;
};

class WAVInput: public FileInput<WAVInputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<WAVInputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new WAVInputStream(s));
      TF_RETURN_IF_ERROR(state.get()->ReadHeader());
    }
    string buffer;
    TF_RETURN_IF_ERROR(state.get()->ReadRecord(record_to_read, record_read, &buffer));
    if (*record_read > 0) {
      Tensor value_tensor(ctx->allocator({}), DT_INT16, {*record_read, state.get()->Channel()});
      memcpy(value_tensor.flat<int16>().data(), buffer.data(), (*record_read) * state.get()->Channel() * sizeof(int16));
      out_tensors->emplace_back(std::move(value_tensor));
    }
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

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(WAVInput, "tensorflow::data::WAVInput");

REGISTER_KERNEL_BUILDER(Name("WAVInput").Device(DEVICE_CPU),
                        FileInputOp<WAVInput>);
REGISTER_KERNEL_BUILDER(Name("WAVDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<WAVInput, WAVInputStream>);
}  // namespace data
}  // namespace tensorflow
