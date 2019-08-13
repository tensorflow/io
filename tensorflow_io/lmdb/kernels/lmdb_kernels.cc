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
#include <sys/types.h>
#include <sys/stat.h>
#include "lmdb.h"

namespace tensorflow {
namespace data {

class LMDBIterable : public IOIterableInterface {
 public:
  LMDBIterable(Env* env)
      : env_(env) {}

  ~LMDBIterable() {
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

  Status Init(const string& input, const void* memory_data, const int64 memory_size, const string& metadata, DataType *dtype, PartialTensorShape* shape) override {
    int status = mdb_env_create(&mdb_env_);
    if (status != MDB_SUCCESS) {
      return errors::InvalidArgument("error on mdb_env_create: ", status);
    }
    int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;

    struct stat source_stat;
    if (stat(input.c_str(), &source_stat) == 0 &&
        (source_stat.st_mode & S_IFREG)) {
      flags |= MDB_NOSUBDIR;
    }
    status = mdb_env_open(mdb_env_, input.c_str(), flags, 0664);
    if (status != MDB_SUCCESS) {
      return errors::InvalidArgument("error on mdb_env_open ", input, ": ", status);
    }
    status = mdb_txn_begin(mdb_env_, nullptr, MDB_RDONLY, &mdb_txn_);
    if (status != MDB_SUCCESS) {
      return errors::InvalidArgument("error on mdb_txn_begin: ", status);
    }
    status = mdb_dbi_open(mdb_txn_, nullptr, 0, &mdb_dbi_);
    if (status != MDB_SUCCESS) {
      return errors::InvalidArgument("error on mdb_dbi_open: ", status);
    }
    status = mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_);
    if (status != MDB_SUCCESS) {
      return errors::InvalidArgument("error on mdb_cursor_open: ", status);
    }
    *dtype = DT_STRING;
    *shape = PartialTensorShape({-1});
    return Status::OK();
  }
  Status Next(const int64 record_to_read, Tensor* value, Tensor* key) override {
    std::vector<string> value_entries, key_entries;
    int64 record_read = 0;
    while (record_read < record_to_read || record_to_read < 0) {
      MDB_val mdb_key;
      MDB_val mdb_value;
      int status = mdb_cursor_get(mdb_cursor_, &mdb_key, &mdb_value, MDB_NEXT);
      if (status != MDB_SUCCESS) {
        break;
      }
      value_entries.emplace_back(std::move(string(static_cast<const char*>(mdb_value.mv_data), mdb_value.mv_size)));
      key_entries.emplace_back(std::move(string(static_cast<const char*>(mdb_key.mv_data), mdb_key.mv_size)));
      record_read++;
    }
    *value = Tensor(DT_STRING, TensorShape({record_read}));
    *key = Tensor(DT_STRING, TensorShape({record_read}));
    for (int64 i = 0; i < record_read; i++) {
      value->flat<string>()(i) = std::move(value_entries[i]);
      key->flat<string>()(i) = std::move(key_entries[i]);
    }
    return Status::OK();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("LMDBIterable[]");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  MDB_env* mdb_env_ GUARDED_BY(mu_) = nullptr;
  MDB_txn* mdb_txn_ GUARDED_BY(mu_) = nullptr;
  MDB_dbi mdb_dbi_ GUARDED_BY(mu_) = 0;

  MDB_cursor* mdb_cursor_ GUARDED_BY(mu_) = nullptr;
};

REGISTER_KERNEL_BUILDER(Name("InitLMDB").Device(DEVICE_CPU),
                        IOInterfaceInitOp<LMDBIterable>);
REGISTER_KERNEL_BUILDER(Name("NextLMDB").Device(DEVICE_CPU),
                        IOIterableNextOp<LMDBIterable>);

//REGISTER_KERNEL_BUILDER(Name("ReadLMDB").Device(DEVICE_CPU),
//                        IOIterableReadOp<LMDBIterable>);

}  // namespace data
}  // namespace tensorflow
