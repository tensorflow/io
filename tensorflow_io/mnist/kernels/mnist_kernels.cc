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

#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/stream.h"

namespace tensorflow {
namespace data {

class MNISTIndexable : public IOIndexableInterface {
 public:
  MNISTIndexable(Env* env)
      : env_(env) {}

  ~MNISTIndexable() {}

  Status Init(const string& input, const void* memory_data, const int64 memory_size, const string& metadata, DataType *dtype, PartialTensorShape* shape) override {
    mutex_lock l(mu_);
    file_.reset(new SizedRandomAccessFile(env_, input, memory_data, memory_size));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    StringPiece result;
    char buffer[8] = {0};
    TF_RETURN_IF_ERROR(file_->Read(0, 8, &result, buffer));
    magic_ =  (((int32)buffer[0] & 0xFF) << 24) | (((int32)buffer[1] & 0xFF) << 16) | (((int32)buffer[2] & 0xFF) << 8) | (((int32)buffer[3] & 0xFF));
    count_ = (((int32)buffer[4] & 0xFF) << 24) | (((int32)buffer[5] & 0xFF) << 16) | (((int32)buffer[6] & 0xFF) << 8) | (((int32)buffer[7] & 0xFF));
    if (magic_ != 0x00000801 && magic_ != 0x00000803) {
      return errors::InvalidArgument("mnist file header must starts with `0x00000801` or `0x00000803`");
    }
    if (magic_ == 0x00000803) {
      TF_RETURN_IF_ERROR(file_->Read(8, 16, &result, buffer));
      rows_ =  (((int32)buffer[0] & 0xFF) << 24) | (((int32)buffer[1] & 0xFF) << 16) | (((int32)buffer[2] & 0xFF) << 8) | (((int32)buffer[3] & 0xFF));
      cols_ = (((int32)buffer[4] & 0xFF) << 24) | (((int32)buffer[5] & 0xFF) << 16) | (((int32)buffer[6] & 0xFF) << 8) | (((int32)buffer[7] & 0xFF));
    } else {
      rows_ = 1;
      cols_ = 1;
    }
    *dtype = DT_UINT8;
    *shape = PartialTensorShape({-1});
    if (magic_ == 0x00000803) {
      *shape = PartialTensorShape({-1, rows_, cols_});
    }

    return Status::OK();
  }
  Status GetItem(const int64 start, const int64 stop, const int64 step, Tensor *value, Tensor* key) override {
    mutex_lock l(mu_);
    int64 offset = (magic_ == 0x00000801) ? (8) : (16);
    offset += start * rows_ * cols_;
    if (magic_ == 0x00000801) {
      *value = Tensor(DT_UINT8, TensorShape({stop - start}));
    } else {
      *value = Tensor(DT_UINT8, TensorShape({stop - start, rows_, cols_}));
    }
    int64 length = (stop - start) * rows_ * cols_;
    StringPiece result;
    Status status = file_->Read(offset, length, &result, (char *)value->flat<uint8>().data());
    if (!(status.ok() || errors::IsOutOfRange(status))) {
      return status;
    }
    if (result.size() != length) {
      return errors::InvalidArgument("corrupted data: offset(", offset, "), length(", length, ")");
    }
    return Status::OK();
  }
  Status Len(int64 *len) override {
    *len = count_;
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("MNISTImageIndexable[]");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  int64 magic_ GUARDED_BY(mu_);
  int64 count_ GUARDED_BY(mu_);
  int64 rows_ GUARDED_BY(mu_);
  int64 cols_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("InitMNIST").Device(DEVICE_CPU),
                        IOInterfaceInitOp<MNISTIndexable>);
REGISTER_KERNEL_BUILDER(Name("GetItemMNIST").Device(DEVICE_CPU),
                        IOIndexableGetItemOp<MNISTIndexable>);
REGISTER_KERNEL_BUILDER(Name("LenMNIST").Device(DEVICE_CPU),
                        IOIndexableLenOp<MNISTIndexable>);
}  // namespace data
}  // namespace tensorflow
