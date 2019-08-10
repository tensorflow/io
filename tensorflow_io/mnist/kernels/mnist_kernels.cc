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

class MNISTLabelIndexable : public IOIndexableInterface {
 public:
  MNISTLabelIndexable(Env* env)
      : env_(env) {}

  ~MNISTLabelIndexable() {
  }

  Status Init(const string& input, const void* memory_data, const int64 memory_size, const string& metadata) override {
    file_.reset(new SizedRandomAccessFile(env_, input, memory_data, memory_size));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    StringPiece result;

    char header[8] = {0};
    TF_RETURN_IF_ERROR(file_->Read(0, 8, &result, header));
    if (header[0] != 0x00 || header[1] != 0x00 || header[2] != 0x08 || header[3] != 0x01) {
      return errors::InvalidArgument("mnist label file header must starts with `0x00000801`");
    }
    count_ = (((int32)header[4] & 0xFF) << 24) | (((int32)header[5] & 0xFF) << 16) | (((int32)header[6] & 0xFF) << 8) | (((int32)header[7] & 0xFF));
    return Status::OK();
  }
  Status GetItem(const int64 start, const int64 stop, std::vector<Tensor>& tensors) override {
    int64 offset = start + 8;
    int64 length = stop - start;
    
    StringPiece result;
    Status status = file_->Read(offset, length, &result, (char *)(tensors[0].flat<uint8>().data()));
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
  Status GetShape(std::vector<TensorShape>& shapes) override {
    shapes.clear();
    shapes.push_back(TensorShape({}));

    return Status::OK();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("MNISTLabelIndexable[]");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  int64 count_ GUARDED_BY(mu_);
};

class MNISTImageIndexable : public IOIndexableInterface {
 public:
  MNISTImageIndexable(Env* env)
      : env_(env) {}

  ~MNISTImageIndexable() {
  }

  Status Init(const string& input, const void* memory_data, const int64 memory_size, const string& metadata) override {
    file_.reset(new SizedRandomAccessFile(env_, input, memory_data, memory_size));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    StringPiece result;

    char header[16] = {0};
    TF_RETURN_IF_ERROR(file_->Read(0, 16, &result, header));
    if (header[0] != 0x00 || header[1] != 0x00 || header[2] != 0x08 || header[3] != 0x03) {
      return errors::InvalidArgument("mnist image file header must starts with `0x00000803`");
    }
    count_ = (((int32)header[4] & 0xFF) << 24) | (((int32)header[5] & 0xFF) << 16) | (((int32)header[6] & 0xFF) << 8) | (((int32)header[7] & 0xFF));
    int64 rows = (((int32)header[8] & 0xFF) << 24) | (((int32)header[9] & 0xFF) << 16) | (((int32)header[10] & 0xFF) << 8) | (((int32)header[11] & 0xFF));
    int64 cols = (((int32)header[12] & 0xFF) << 24) | (((int32)header[13] & 0xFF) << 16) | (((int32)header[14] & 0xFF) << 8) | (((int32)header[15] & 0xFF));

    shape_ = TensorShape({rows, cols});

    return Status::OK();
  }
  Status GetItem(const int64 start, const int64 stop, std::vector<Tensor>& tensors) override {
    int64 offset = start + 16;
    int64 length = (stop - start) * shape_.dim_size(0) * shape_.dim_size(1);
    
    StringPiece result;
    Status status = file_->Read(offset, length, &result, (char *)(tensors[0].flat<uint8>().data()));
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
  Status GetShape(std::vector<TensorShape>& shapes) override {
    shapes.clear();
    shapes.push_back(shape_);

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
  int64 count_ GUARDED_BY(mu_);
  TensorShape shape_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("ReadMNISTLabel").Device(DEVICE_CPU),
                        IOIndexableReadOp<MNISTLabelIndexable>);
REGISTER_KERNEL_BUILDER(Name("ReadMNISTImage").Device(DEVICE_CPU),
                        IOIndexableReadOp<MNISTImageIndexable>);

}  // namespace data
}  // namespace tensorflow
