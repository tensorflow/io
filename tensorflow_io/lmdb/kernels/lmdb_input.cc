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
#include "lmdb.h"

namespace tensorflow {
namespace data {

class LMDBInputStream{
public:
  explicit LMDBInputStream(const string& filename)
    : mdb_status_(MDB_SUCCESS)
    , mdb_env_(nullptr)
    , mdb_txn_(nullptr)
    , mdb_dbi_(0)
    , mdb_cursor_(nullptr) {
    mdb_status_ = mdb_env_create(&mdb_env_);
    if (mdb_status_ != MDB_SUCCESS) {
      return;
    }
    int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;

    struct stat source_stat;
    if (stat(filename.c_str(), &source_stat) == 0 &&
        (source_stat.st_mode & S_IFREG)) {
      flags |= MDB_NOSUBDIR;
    }
    mdb_status_ = mdb_env_open(mdb_env_, filename.c_str(), flags, 0664);
    if (mdb_status_ != MDB_SUCCESS) {
      return;
    }
    mdb_status_ = mdb_txn_begin(mdb_env_, nullptr, MDB_RDONLY, &mdb_txn_);
    if (mdb_status_ != MDB_SUCCESS) {
      return;
    }
    mdb_status_ = mdb_dbi_open(mdb_txn_, nullptr, 0, &mdb_dbi_);
    if (mdb_status_ != MDB_SUCCESS) {
      return;
    }
    mdb_status_ = mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_);
    if (mdb_status_ != MDB_SUCCESS) {
      return;
    }
    mdb_status_ = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
    if (mdb_status_ != MDB_SUCCESS && mdb_status_ != MDB_NOTFOUND) {
      return;
    }
    if (mdb_status_ == MDB_NOTFOUND) {
      // empty data, move on.
    }
    return;
  }
  ~LMDBInputStream() {
    if (mdb_env_ != nullptr) {
      if (mdb_cursor_) {
        mdb_cursor_close(mdb_cursor_);
        mdb_cursor_ = nullptr;
      }
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_txn_abort(mdb_txn_);
      mdb_env_close(mdb_env_);
      mdb_txn_ = nullptr;
      mdb_dbi_ = 0;
      mdb_env_ = nullptr;
    }
  }
  Status ReadRecord(string* key, string* val) {
    if (mdb_status_ != MDB_SUCCESS) {
      if (mdb_status_ == MDB_NOTFOUND) {
         return errors::OutOfRange("EOF reached");
      }
      return errors::InvalidArgument(mdb_strerror(mdb_status_));
    }
    *key = std::move(string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size));
    *val = std::move(string(static_cast<const char*>(mdb_value_.mv_data), mdb_value_.mv_size));
    mdb_status_ = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT);
    return Status::OK();
  }
private:
  int mdb_status_ = MDB_SUCCESS;
  MDB_env* mdb_env_ = nullptr;
  MDB_txn* mdb_txn_ = nullptr;
  MDB_dbi mdb_dbi_ = 0;

  MDB_cursor* mdb_cursor_ = nullptr;
  MDB_val mdb_key_;
  MDB_val mdb_value_;
};

class LMDBInput: public FileInput<LMDBInputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<LMDBInputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      // LMDB does not support compression or any file system so
      // using filename() (instead of stream/s) here is fine.
      state.reset(new LMDBInputStream(filename()));
    }
    Tensor key_tensor(ctx->allocator({}), DT_STRING, {record_to_read});
    Tensor val_tensor(ctx->allocator({}), DT_STRING, {record_to_read});
    while ((*record_read) < record_to_read) {
      string key, val;
      Status status = state.get()->ReadRecord(&key, &val);
      if (!(status.ok() || errors::IsOutOfRange(status))) {
        return status;
      }
      if (!status.ok()) {
        break;
      }
      key_tensor.flat<string>()(*record_read) = std::move(key);
      val_tensor.flat<string>()(*record_read) = std::move(val);
      (*record_read)++;
    }
    if (*record_read > 0) {
      out_tensors->emplace_back(std::move(key_tensor));
      out_tensors->emplace_back(std::move(val_tensor));
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

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(LMDBInput, "tensorflow::data::LMDBInput");

REGISTER_KERNEL_BUILDER(Name("IO>LMDBInput").Device(DEVICE_CPU),
                        FileInputOp<LMDBInput>);
REGISTER_KERNEL_BUILDER(Name("IO>LMDBDatasetV2").Device(DEVICE_CPU),
                        FileInputDatasetOp<LMDBInput, LMDBInputStream>);

}  // namespace data
}  // namespace tensorflow
