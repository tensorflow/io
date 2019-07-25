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
        H5::DataSet dataset = file_->openDataSet(H5std_string(columns_[i]));
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        absl::InlinedVector<hsize_t, 4> dims(rank);
        dataspace.getSimpleExtentDims(dims.data());
        dataset_.emplace_back(dataset);
        dataspace_.emplace_back(dataspace);
        dims_.emplace_back(dims);

        // Make sure first dimension remains the same
        if (i == 0) {
          count_ = dims[0];
        } else if (count_ != dims[0]) {
          // Maybe we should fill in blanks?
          return errors::InvalidArgument("dataset ", columns_[i], " has uneven count ", dims[0], " with others ", count_);
        }
      } catch(H5::FileIException e){
        return errors::InvalidArgument("unable to open dataset ", columns_[i], ": ", e.getCDetailMsg());
      }
    }
    return Status::OK();
  }
  Status ReadRecord(IteratorContext* ctx, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) {
    if (index_ + record_to_read > count_) {
      record_to_read = count_ - index_;
    }
    out_tensors->clear();
    if (record_to_read > 0) {
      for (size_t i = 0; i < columns_.size(); i++) {
        absl::InlinedVector<hsize_t, 4> dims = dims_[i];
        dims[0] = record_to_read;
        H5::DataSpace memoryspace(dims.size(), dims.data());
        absl::InlinedVector<hsize_t, 4> start(dims_[i].size(), 0);
        start[0] = index_;
        dataspace_[i].selectHyperslab(H5S_SELECT_SET, dims.data(), start.data());

        absl::InlinedVector<int64, 4> shape_dims(dims_[i].size());
        for (size_t ii = 0; ii < dims_[i].size(); ii++) {
          shape_dims[ii] = dims_[i][ii];
        }
        shape_dims[0] = record_to_read;
        TensorShape shape(shape_dims);


        H5::DataType data_type = dataset_[i].getDataType();
        hid_t native_type = H5Tget_native_type(data_type.getId(), H5T_DIR_ASCEND);
        if (H5Tequal(native_type, H5T_NATIVE_INT)) {
          Tensor tensor(ctx->allocator({}), DT_INT32, shape);
          dataset_[i].read(tensor.flat<int32>().data(), H5::PredType::NATIVE_INT, memoryspace, dataspace_[i]);
          out_tensors->emplace_back(std::move(tensor));
        } else if (H5Tequal(native_type, H5T_NATIVE_UINT32)) {
          Tensor tensor(ctx->allocator({}), DT_UINT32, shape);
          dataset_[i].read(tensor.flat<uint32>().data(), H5::PredType::NATIVE_UINT32, memoryspace, dataspace_[i]);
          out_tensors->emplace_back(std::move(tensor));
        }else if (H5Tequal(native_type, H5T_NATIVE_LONG)) {
          Tensor tensor(ctx->allocator({}), DT_INT64, shape);
          dataset_[i].read(tensor.flat<int64>().data(), H5::PredType::NATIVE_LONG, memoryspace, dataspace_[i]);
          out_tensors->emplace_back(std::move(tensor));
        } else if (H5Tequal(native_type, H5T_NATIVE_FLOAT)) {
          Tensor tensor(ctx->allocator({}), DT_FLOAT, shape);
          dataset_[i].read(tensor.flat<float>().data(), H5::PredType::NATIVE_FLOAT, memoryspace, dataspace_[i]);
          out_tensors->emplace_back(std::move(tensor));
        } else if (H5Tequal(native_type, H5T_NATIVE_DOUBLE)) {
          Tensor tensor(ctx->allocator({}), DT_DOUBLE, shape);
          dataset_[i].read(tensor.flat<double>().data(), H5::PredType::NATIVE_DOUBLE, memoryspace, dataspace_[i]);
          out_tensors->emplace_back(std::move(tensor));
        } else {
          return errors::Unimplemented("data type not supported yet: ", dataset_[i].getTypeClass());
        }
      }
      *record_read = record_to_read;
      index_ += record_to_read;
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
  std::vector<H5::DataSet> dataset_;
  std::vector<H5::DataSpace> dataspace_;
  std::vector<absl::InlinedVector<hsize_t, 4>> dims_;
  int64 count_ = -1;
  int64 index_ = 0;
};

class HDF5Input: public FileInput<HDF5InputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<HDF5InputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new HDF5InputStream(s, columns()));
      TF_RETURN_IF_ERROR(state.get()->Open());
    }
    return state.get()->ReadRecord(ctx, record_to_read, record_read, out_tensors);
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
