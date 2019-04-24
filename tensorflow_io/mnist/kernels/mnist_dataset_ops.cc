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

namespace tensorflow {
namespace data {

class MNISTImageInput: public FileInput<int64> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<int64>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new int64(0));
      TF_RETURN_IF_ERROR(s->SkipNBytes(16));
    }
    string buffer;
    Status status = ReadInputStream(s, (rows_ * cols_), record_to_read, &buffer, record_read);
    if (!(status.ok() || errors::IsOutOfRange(status))) {
      return status;
    }
    (*(state.get())) += *record_read;
    if (*record_read > 0) {
      Tensor value_tensor(ctx->allocator({}), DT_UINT8, {*record_read, rows_, cols_});
      memcpy(value_tensor.flat<uint8>().data(), buffer.data(), (*record_read) * rows_ * cols_);
      out_tensors->emplace_back(std::move(value_tensor));
    }
    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface* s) override {
    string header;
    TF_RETURN_IF_ERROR(s->ReadNBytes(16, &header));
    if (header[0] != 0x00 || header[1] != 0x00 || header[2] != 0x08 || header[3] != 0x03) {
      return errors::InvalidArgument("mnist image file header must starts with `0x00000803`");
    }
    size_ = (((int32)header[4] & 0xFF) << 24) | (((int32)header[5] & 0xFF) << 16) | (((int32)header[6] & 0xFF) << 8) | (((int32)header[7] & 0xFF));
    rows_ = (((int32)header[8] & 0xFF) << 24) | (((int32)header[9] & 0xFF) << 16) | (((int32)header[10] & 0xFF) << 8) | (((int32)header[11] & 0xFF));
    cols_ = (((int32)header[12] & 0xFF) << 24) | (((int32)header[13] & 0xFF) << 16) | (((int32)header[14] & 0xFF) << 8) | (((int32)header[15] & 0xFF));
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
    data->tensors_.emplace_back(Tensor(DT_INT64, TensorShape({})));
    data->tensors_.emplace_back(Tensor(DT_INT64, TensorShape({})));
    data->tensors_.emplace_back(Tensor(DT_INT64, TensorShape({})));
    data->tensors_[3].scalar<int64>()() = size_;
    data->tensors_[4].scalar<int64>()() = rows_;
    data->tensors_[5].scalar<int64>()() = cols_;
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    size_ = data.tensors(3).scalar<int64>()();
    rows_ = data.tensors(4).scalar<int64>()();
    cols_ = data.tensors(5).scalar<int64>()();
    return true;
  }
 protected:
  int64 size_;
  int64 rows_;
  int64 cols_;
};
class MNISTLabelInput: public FileInput<int64> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<int64>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new int64(0));
      TF_RETURN_IF_ERROR(s->SkipNBytes(8));
    }
    string buffer;
    TF_RETURN_IF_ERROR(ReadInputStream(s, 1, record_to_read, &buffer, record_read));
    (*(state.get())) += *record_read;
    if (*record_read > 0) {
      Tensor value_tensor(ctx->allocator({}), DT_UINT8, {*record_read});
      memcpy(value_tensor.flat<uint8>().data(), buffer.data(), (*record_read));
      out_tensors->emplace_back(std::move(value_tensor));
    }
    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface* s) override {
    string header;
    TF_RETURN_IF_ERROR(s->ReadNBytes(8, &header));
    if (header[0] != 0x00 || header[1] != 0x00 || header[2] != 0x08 || header[3] != 0x01) {
      return errors::InvalidArgument("mnist label file header must starts with `0x00000801`");
    }
    size_ = (((int32)header[4] & 0xFF) << 24) | (((int32)header[5] & 0xFF) << 16) | (((int32)header[6] & 0xFF) << 8) | (((int32)header[7] & 0xFF));
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
    data->tensors_.emplace_back(Tensor(DT_INT64, TensorShape({})));
    data->tensors_[3].scalar<int64>()() = size_;
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    size_ = data.tensors(3).scalar<int64>()();
    return true;
  }
 protected:
  int64 size_;
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(MNISTLabelInput, "tensorflow::data::MNISTLabelInput");
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(MNISTImageInput, "tensorflow::data::MNISTImageInput");

REGISTER_KERNEL_BUILDER(Name("MNISTLabelInput").Device(DEVICE_CPU),
                        FileInputOp<MNISTLabelInput>);
REGISTER_KERNEL_BUILDER(Name("MNISTImageInput").Device(DEVICE_CPU),
                        FileInputOp<MNISTImageInput>);
REGISTER_KERNEL_BUILDER(Name("MNISTLabelDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<MNISTLabelInput, int64>);
REGISTER_KERNEL_BUILDER(Name("MNISTImageDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<MNISTImageInput, int64>);
}  // namespace data
}  // namespace tensorflow
