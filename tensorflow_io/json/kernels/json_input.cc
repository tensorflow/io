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

#include <iostream>
#include <fstream>
#include "kernels/dataset_ops.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "include/json/json.h"

namespace tensorflow {
namespace data {

class JSONInputStream {
public:
  explicit JSONInputStream(const string& filename) {
      Env* env = Env::Default();
      uint64 size = 0;
      Status status = env->GetFileSize(filename, &size);
      if (status.ok()) {
        std::unique_ptr<tensorflow::RandomAccessFile> file;
        status = env->NewRandomAccessFile(filename, &file);
        if (status.ok()) {
          StringPiece result;
          buffer_memory_.resize(size);
          status = file->Read(0, size, &result, &buffer_memory_[0]);
        }
      }
    }
   
  ~JSONInputStream() {}
  Status Open(){
    if (reader_.parse(buffer_memory_, records_)) {
      return Status::OK();
    }
    return errors::InvalidArgument("JSON parsing error: ", reader_.getFormattedErrorMessages());
  }

bool ReadRecord(Json::Value& record){
  if (index_ < records_.size()){
    record = records_[index_];
    index_++;
    return true;
  }
  return false;
}

private:
  string buffer_memory_;
  string filename_;
  Json::Reader reader_;
  Json::Value records_;
  Json::ArrayIndex index_ = 0;
};

class JSONInput: public FileInput<JSONInputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<JSONInputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new JSONInputStream(filename()));
      TF_RETURN_IF_ERROR(state.get()->Open());
    }

    Json::Value record;
    while ((*record_read) < record_to_read && state.get()->ReadRecord(record)) {
      if(*record_read == 0){
        out_tensors->clear();
        //allocate enough space for Tensor
        for (size_t i = 0; i < columns().size(); i++){
          const string& column = columns()[i];
          const Json::Value& val = record[column];
          DataType dtype;
          switch (val.type()){
          case Json::intValue:
            dtype = DT_INT64;
            break;
          case Json::uintValue:
            dtype = DT_UINT64;
            break;
          case Json::realValue:
            dtype = DT_DOUBLE;
            break;
          case Json::stringValue:
            dtype = DT_STRING;
            break;
          case Json::booleanValue:
            dtype = DT_BOOL;
            break;
          //Currently Json::nullValue, Json::arrayValue, and Json::objectValue are not supported.
          default:
            return errors::InvalidArgument("Unsupported data type: ", val.type());
          }
          Tensor tensor(ctx->allocator({}), dtype, {record_to_read});
          out_tensors->emplace_back(std::move(tensor));
        }
      }
      for (size_t i = 0; i < columns().size(); i++) {
        const string& column = columns()[i];
        const Json::Value& val = record[column];
        switch (val.type()){
        case Json::intValue:
          ((*out_tensors)[i]).flat<int64>()(*record_read) = val.asInt64();
          break;
        case Json::uintValue:
          ((*out_tensors)[i]).flat<uint64>()(*record_read) = val.asUInt64();
          break;
        case Json::realValue:
          ((*out_tensors)[i]).flat<double>()(*record_read) = val.asDouble();
          break;
        case Json::stringValue:
          ((*out_tensors)[i]).flat<string>()(*record_read) = val.asString();
          break;
        case Json::booleanValue:
          ((*out_tensors)[i]).flat<bool>()(*record_read) = val.asBool();
          break;
        //Currently Json::nullValue, Json::arrayValue, and Json::objectValue are not supported.
        default:
          return errors::InvalidArgument("Unsupported data type: ", val.type());
        }
      }
      (*record_read)++;
    }

    //Slice if needed
    if (*record_read < record_to_read) {
      if (*record_read == 0) {
        out_tensors->clear();
      }
      for (size_t i = 0; i < out_tensors->size(); i++) {
        Tensor tensor = (*out_tensors)[i].Slice(0, *record_read);
        (*out_tensors)[i] = std::move(tensor);
      }
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

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(JSONInput, "tensorflow::data::JSONInput");

REGISTER_KERNEL_BUILDER(Name("JSONInput").Device(DEVICE_CPU),
                        FileInputOp<JSONInput>);
REGISTER_KERNEL_BUILDER(Name("JSONDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<JSONInput, JSONInputStream>); 
}  // namespace data
}  // namespace tensorflow
