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
#include <hdf5.h>
#include <hdf5_hl.h>
#include <H5Cpp.h>

namespace tensorflow {
namespace data {

class HDF5InputStream{
public:
  explicit HDF5InputStream(io::InputStreamInterface* s, const std::vector<string>& columns)
    : columns_(columns)
    , input_stream_(nullptr)
    , buffered_stream_(nullptr)
    , file_(nullptr) {
    input_stream_ = dynamic_cast<SizedRandomAccessInputStreamInterface*>(s);
    if (input_stream_ == nullptr) {
      buffered_stream_.reset(new SizedRandomAccessBufferedStream(s));
      input_stream_ = buffered_stream_.get();
    }
  }
  ~HDF5InputStream() {
    H5Fclose(file_image_);
    file_.reset(nullptr);
    buffered_stream_.reset(nullptr);
  }
  Status Open() {
    uint64 size = 0;
    TF_RETURN_IF_ERROR(input_stream_->GetFileSize(&size));
    buffer_.resize(size);
    StringPiece result;
    TF_RETURN_IF_ERROR(input_stream_->Read(0, size, &result, &buffer_[0]));
    if (result.size() != size) {
      return errors::InvalidArgument("unable to read enough data from file");
    }
    file_image_ = H5LTopen_file_image((void *)buffer_.data(), size, H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_DONT_RELEASE);
    file_.reset(new H5::H5File());
    file_.get()->setId(file_image_);
    // TODO: replace boilerplate
    for (size_t i = 0; i < columns_.size(); i++) {
      try {
        H5::DataSet data_set(file_->openDataSet(H5std_string(columns_[i])));
      } catch(H5::FileIException e){
        return errors::InvalidArgument("unable to open dataset ", columns_[i], ": ", e.getCDetailMsg());
      }
    }
    return Status::OK();
  }
private:
  std::vector<string> columns_;
  SizedRandomAccessInputStreamInterface* input_stream_;
  std::unique_ptr<SizedRandomAccessBufferedStream> buffered_stream_;
  string buffer_;
  std::unique_ptr<H5::H5File> file_;
  hid_t file_image_;
};

class HDF5Input: public FileInput<HDF5InputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<HDF5InputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new HDF5InputStream(s, columns()));
      TF_RETURN_IF_ERROR(state.get()->Open());
    }
    return errors::Unimplemented("HDF5 is currently not supported");
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

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(HDF5Input, "tensorflow::data::HDF5Input");

REGISTER_KERNEL_BUILDER(Name("HDF5Input").Device(DEVICE_CPU),
                        FileInputOp<HDF5Input>);
REGISTER_KERNEL_BUILDER(Name("HDF5Dataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<HDF5Input, HDF5InputStream>);
}  // namespace data
}  // namespace tensorflow
