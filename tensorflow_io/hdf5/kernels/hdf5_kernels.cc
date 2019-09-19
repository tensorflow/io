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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/stream.h"

#include <hdf5.h>
#include <hdf5_hl.h>
#include <H5Cpp.h>

namespace tensorflow {
namespace data {
namespace {

class HDF5FileImage {
 public:
  HDF5FileImage(Env* env, const string& filename, const string& optional_memory)
  : filename_(filename)
  , optional_memory_(optional_memory)
  , file_(nullptr) {
    if (optional_memory.size() != 0) {
      file_image_ = H5LTopen_file_image((void *)optional_memory_.data(), optional_memory_.size(), H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_DONT_RELEASE);
      file_.reset(new H5::H5File());
      file_.get()->setId(file_image_);
    } else if (filename.find("://") == string::npos) {
      file_.reset(new H5::H5File(filename, H5F_ACC_RDONLY));
    } else {
      uint64 size = 0;
      Status status = env->GetFileSize(filename, &size);
      if (status.ok()) {
        std::unique_ptr<tensorflow::RandomAccessFile> file;
        status = env->NewRandomAccessFile(filename, &file);
        if (status.ok()) {
          StringPiece result;
          buffer_memory_.resize(size);
          status = file->Read(0, size, &result, &buffer_memory_[0]);
          if (status.ok()) {
            file_image_ = H5LTopen_file_image((void *)buffer_memory_.data(), buffer_memory_.size(), H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_DONT_RELEASE);
            file_.reset(new H5::H5File());
            file_.get()->setId(file_image_);
          }
        }
      }
    }
  }

  virtual ~HDF5FileImage() {
    if (file_image_ != 0) {
      H5Fclose(file_image_);
    }
    file_.reset(nullptr);
  }

  H5::H5File *GetFile() const {
    return file_.get();
  }


 private:
  string filename_;
  const string& optional_memory_;
  string buffer_memory_;
  std::unique_ptr<H5::H5File> file_;
  hid_t file_image_ = 0;
};

class HDF5Iterate {
public:
  HDF5Iterate(haddr_t root)
  : parent_(root) {
    groups_[root] = "";
  }
  ~HDF5Iterate() {}

  static herr_t Iterate(hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data) {
    HDF5Iterate *p = (HDF5Iterate *)operator_data;

    H5O_info_t iteminfo;
    herr_t err = H5Oget_info_by_name (loc_id, name, &iteminfo, H5P_DEFAULT);

    switch (iteminfo.type) {
    case H5O_TYPE_GROUP:
      if (p->groups_.find(iteminfo.addr) == p->groups_.end()) {
        haddr_t parent = p->parent_;
        p->groups_[iteminfo.addr] = p->groups_[parent] + "/" + name;
        p->parent_ = iteminfo.addr;
        err = H5Literate_by_name(loc_id, name, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, HDF5Iterate::Iterate, operator_data, H5P_DEFAULT);
        p->parent_ = parent;
      }
      break;
    case H5O_TYPE_DATASET: {
        string dataset = p->groups_[p->parent_] + "/" + name;
        p->datasets_.emplace_back(dataset);
      }
      break;
    case H5O_TYPE_NAMED_DATATYPE:
      break;
    default:
      break;
    }
    return err;
  }
 
  std::vector<string> datasets_;
  std::unordered_map<haddr_t, string> groups_;
  haddr_t parent_;
};

class ListHDF5DatasetsOp : public OpKernel {
 public:
  explicit ListHDF5DatasetsOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();

    HDF5FileImage file_image(env_, filename, memory);
    H5::H5File *file = file_image.GetFile();
    OP_REQUIRES(context, file != nullptr, errors::InvalidArgument("unable to open hdf5 file: ", filename));

    H5O_info_t info;
    file->getObjinfo(info);

    HDF5Iterate data(info.addr);

    herr_t err = H5Literate (file->getId(), H5_INDEX_NAME, H5_ITER_NATIVE, NULL, HDF5Iterate::Iterate, (void *)&data);

    std::vector<string> datasets;
    std::vector<string> dtypes;
    std::vector<absl::InlinedVector<hsize_t, 4>> shapes;
    datasets.reserve(data.datasets_.size());
    dtypes.reserve(data.datasets_.size());
    shapes.reserve(data.datasets_.size());
    int maxrank = 0;
    for (size_t i = 0; i < data.datasets_.size(); i++) {
      string dataset = data.datasets_[i];
      string dtype = "";
      H5::DataSet data_set = file->openDataSet(dataset);

      H5::DataSpace data_space = data_set.getSpace();
      int rank = data_space.getSimpleExtentNdims();
      absl::InlinedVector<hsize_t, 4> dims(rank);
      data_space.getSimpleExtentDims(dims.data());

      maxrank = rank < maxrank ? maxrank : rank;

      H5::DataType data_type = data_set.getDataType();
      hid_t native_type = H5Tget_native_type(data_type.getId(), H5T_DIR_ASCEND);
      if (H5Tequal(native_type, H5T_NATIVE_INT8)) {
        dtype = "int8";
      } else if (H5Tequal(native_type, H5T_NATIVE_UINT8)) {
        dtype = "uint8";
      } else if (H5Tequal(native_type, H5T_NATIVE_INT)) {
        dtype = "int32";
      } else if (H5Tequal(native_type, H5T_NATIVE_UINT32)) {
        dtype = "uint32";
      } else if (H5Tequal(native_type, H5T_NATIVE_LONG)) {
        dtype = "int64";
      } else if (H5Tequal(native_type, H5T_NATIVE_FLOAT)) {
        dtype = "float";
      } else if (H5Tequal(native_type, H5T_NATIVE_DOUBLE)) {
        dtype = "double";
      } else {
        continue;
      }
      datasets.emplace_back(dataset);
      dtypes.emplace_back(dtype);
      shapes.emplace_back(dims);
    }
 
    TensorShape output_shape = filename_tensor.shape();
    output_shape.AddDim(datasets.size());

    Tensor* datasets_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &datasets_tensor));
    Tensor* dtypes_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &dtypes_tensor));

    for (size_t i = 0; i < datasets.size(); i++) {
      datasets_tensor->flat<string>()(i) = datasets[i];
      dtypes_tensor->flat<string>()(i) = dtypes[i];
    }

    output_shape.AddDim(maxrank);

    Tensor* shapes_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape, &shapes_tensor));
    for (size_t i = 0; i < shapes.size(); i++) {
      for (size_t j = 0; j < shapes[i].size(); j++) {
        shapes_tensor->flat<int64>()(i * maxrank + j) = shapes[i][j];
      }
      for (size_t j = shapes[i].size(); j < maxrank; j++) {
        shapes_tensor->flat<int64>()(i * maxrank + j) = -1;
      }
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class ReadHDF5Op : public OpKernel {
 public:
  explicit ReadHDF5Op(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string& filename = filename_tensor.scalar<string>()();

    const Tensor& dataset_tensor = context->input(1);
    const string& dataset = dataset_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(2);
    const string& memory = memory_tensor.scalar<string>()();

    const Tensor& start_tensor = context->input(3);

    const Tensor& stop_tensor = context->input(4);

    HDF5FileImage file_image(env_, filename, memory);
    H5::H5File *file = file_image.GetFile();
    OP_REQUIRES(context, file != nullptr, errors::InvalidArgument("unable to open hdf5 file: ", filename));
    try {
      H5::DataSet data_set = file->openDataSet(dataset);

      H5::DataSpace data_space = data_set.getSpace();
      int rank = data_space.getSimpleExtentNdims();
      absl::InlinedVector<hsize_t, 4> dims(rank);
      data_space.getSimpleExtentDims(dims.data());

    std::vector<int64> start(dims.size(), 0);
    std::vector<int64> stop(dims.size(), -1);
    for (size_t i = 0; i < start_tensor.NumElements(); i++) {
      start[i] = start_tensor.flat<int64>()(i);
    }
    for (size_t i = 0; i < stop_tensor.NumElements(); i++) {
      stop[i] = stop_tensor.flat<int64>()(i);
    }
    for (size_t i = 0; i < stop.size(); i++) {
      if (stop[i] < 0) {
        stop[i] = dims[i];
      }
    }

    // Find the border of the dims start
    absl::InlinedVector<hsize_t, 4> dims_start(dims.size(), 0);
    for (int64 i = 0; i < dims_start.size(); i++) {
      dims_start[i] = (start[i] < dims[i]) ? (start[i]) : (dims[i]);
    }
    // Find the border of the dims final
    absl::InlinedVector<hsize_t, 4> dims_final(dims);
    for (int64 i = 0; i < dims_final.size(); i++) {
      dims_final[i] = (stop[i] < dims[i]) ? (stop[i]) : (dims[i]);
    }
    // Find the area of the dims = [start...final]
    absl::InlinedVector<int64, 4> dims_shape(dims.size());
    for (int64 i = 0; i < dims_shape.size(); i++) {
      dims[i] = (dims_final[i] > dims_start[i]) ?  (dims_final[i] - dims_start[i]) : 0;
      dims_shape[i] = dims[i];
    }

    TensorShape output_shape(dims_shape);

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    // Return with zero elements
    for (int64 i = 0; i < dims_shape.size(); i++) {
      if (dims_shape[0] == 0) {
        return;
      }
    }

    H5::DataSpace memory_space(dims.size(), dims.data());

    data_space.selectHyperslab(H5S_SELECT_SET, dims.data(), dims_start.data());

    H5::DataType data_type = data_set.getDataType();
    hid_t native_type = H5Tget_native_type(data_type.getId(), H5T_DIR_ASCEND);
    if (H5Tequal(native_type, H5T_NATIVE_INT8)) {
      data_set.read(output_tensor->flat<int8>().data(), H5::PredType::NATIVE_INT8, memory_space, data_space);
    } else if (H5Tequal(native_type, H5T_NATIVE_UINT8)) {
      data_set.read(output_tensor->flat<uint8>().data(), H5::PredType::NATIVE_UINT8, memory_space, data_space);
    } else if (H5Tequal(native_type, H5T_NATIVE_INT)) {
      data_set.read(output_tensor->flat<int32>().data(), H5::PredType::NATIVE_INT, memory_space, data_space);
    } else if (H5Tequal(native_type, H5T_NATIVE_UINT32)) {
      data_set.read(output_tensor->flat<uint32>().data(), H5::PredType::NATIVE_UINT32, memory_space, data_space);
    } else if (H5Tequal(native_type, H5T_NATIVE_LONG)) {
      data_set.read(output_tensor->flat<int64>().data(), H5::PredType::NATIVE_LONG, memory_space, data_space);
    } else if (H5Tequal(native_type, H5T_NATIVE_FLOAT)) {
      data_set.read(output_tensor->flat<float>().data(), H5::PredType::NATIVE_FLOAT, memory_space, data_space);
    } else if (H5Tequal(native_type, H5T_NATIVE_DOUBLE)) {
      data_set.read(output_tensor->flat<double>().data(), H5::PredType::NATIVE_DOUBLE, memory_space, data_space);
    } else {
      OP_REQUIRES(context, false, errors::Unimplemented("data type not supported yet: ", data_set.getTypeClass()));
    }
    } catch(H5::FileIException e){
      OP_REQUIRES(context, false, errors::InvalidArgument("unable to open dataset", e.getCDetailMsg()));
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("ListHDF5Datasets").Device(DEVICE_CPU),
                        ListHDF5DatasetsOp);
REGISTER_KERNEL_BUILDER(Name("ReadHDF5").Device(DEVICE_CPU),
                        ReadHDF5Op);


}  // namespace


class HDF5Indexable : public IOIndexableInterface {
 public:
  HDF5Indexable(Env* env)
  : env_(env) {}

  ~HDF5Indexable() {}
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 filename is not supported");
    }
    const string& filename = input[0];
    file_.reset(new SizedRandomAccessFile(env_, filename, memory_data, memory_size));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    file_image_.reset(new HDF5FileImage(env_, filename, ""));
    H5::H5File *file = file_image_->GetFile();
    if (file == nullptr) {
      return errors::InvalidArgument("unable to open hdf5 file: ", filename);
    }

    H5O_info_t info;
    file->getObjinfo(info);
    HDF5Iterate data(info.addr);
    herr_t err = H5Literate(file->getId(), H5_INDEX_NAME, H5_ITER_NATIVE, NULL, HDF5Iterate::Iterate, (void *)&data);
    for (size_t i = 0; i < data.datasets_.size(); i++) {
      columns_.emplace_back(data.datasets_[i]);
      columns_index_[data.datasets_[i]] = i;
    }

    for (size_t i = 0; i < columns_.size(); i++) {
      ::tensorflow::DataType dtype;
      string dataset = columns_[i];
      H5::DataSet data_set = file->openDataSet(dataset);

      H5::DataSpace data_space = data_set.getSpace();
      int rank = data_space.getSimpleExtentNdims();
      absl::InlinedVector<hsize_t, 4> dims(rank);
      data_space.getSimpleExtentDims(dims.data());

      H5::DataType data_type = data_set.getDataType();
      hid_t native_type = H5Tget_native_type(data_type.getId(), H5T_DIR_ASCEND);
      if (H5Tequal(native_type, H5T_NATIVE_INT8)) {
        dtype = DT_INT8;
      } else if (H5Tequal(native_type, H5T_NATIVE_UINT8)) {
        dtype = DT_UINT8;
      } else if (H5Tequal(native_type, H5T_NATIVE_INT)) {
        dtype = DT_INT32;
      } else if (H5Tequal(native_type, H5T_NATIVE_UINT32)) {
        dtype = DT_UINT32;
      } else if (H5Tequal(native_type, H5T_NATIVE_LONG)) {
        dtype = DT_INT64;
      } else if (H5Tequal(native_type, H5T_NATIVE_FLOAT)) {
        dtype = DT_FLOAT;
      } else if (H5Tequal(native_type, H5T_NATIVE_DOUBLE)) {
        dtype = DT_DOUBLE;
      } else {
        return errors::InvalidArgument("unsupported data type: ", native_type);
      }
      dtypes_.emplace_back(dtype);
      absl::InlinedVector<int64, 4> shape_dims(rank);
      for (int r = 0; r < rank; r++) {
        shape_dims[r] = dims[r];
      }
      shapes_.emplace_back(TensorShape(shape_dims));
    }
    return Status::OK();
  }
  Status Components(Tensor* components) override {
    *components = Tensor(DT_STRING, TensorShape({static_cast<int64>(columns_.size())}));
    for (size_t i = 0; i < columns_.size(); i++) {
      components->flat<string>()(i) = columns_[i];
    }
    return Status::OK();
  }
  Status Spec(const Tensor& component, PartialTensorShape* shape, DataType* dtype, bool label) override {
    const int64 column_index = columns_index_[component.scalar<string>()()];
    *shape = shapes_[column_index];
    *dtype = dtypes_[column_index];
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop, const Tensor& component, Tensor* value, Tensor* label) override {
    const string& column = component.scalar<string>()();

    H5::H5File *file = file_image_->GetFile();
    try {
      H5::DataSet data_set = file->openDataSet(column);
      H5::DataSpace data_space = data_set.getSpace();

      int rank = data_space.getSimpleExtentNdims();
      absl::InlinedVector<hsize_t, 4> dims(rank);
      data_space.getSimpleExtentDims(dims.data());

      if (start > dims[0] || stop > dims[0]) {
        return errors::InvalidArgument("dataset ", column, " selection is out of boundary");
      }
      // Find the border of the dims start and dims
      absl::InlinedVector<hsize_t, 4> dims_start(dims.size(), 0);
      dims_start[0] = start;
      dims[0] = stop - start;

      H5::DataSpace memory_space(dims.size(), dims.data());

      data_space.selectHyperslab(H5S_SELECT_SET, dims.data(), dims_start.data());

      H5::DataType data_type = data_set.getDataType();
      hid_t native_type = H5Tget_native_type(data_type.getId(), H5T_DIR_ASCEND);
      if (H5Tequal(native_type, H5T_NATIVE_INT8)) {
        data_set.read(value->flat<int8>().data(), H5::PredType::NATIVE_INT8, memory_space, data_space);
      } else if (H5Tequal(native_type, H5T_NATIVE_UINT8)) {
        data_set.read(value->flat<uint8>().data(), H5::PredType::NATIVE_UINT8, memory_space, data_space);
      } else if (H5Tequal(native_type, H5T_NATIVE_INT)) {
        data_set.read(value->flat<int32>().data(), H5::PredType::NATIVE_INT, memory_space, data_space);
      } else if (H5Tequal(native_type, H5T_NATIVE_UINT32)) {
        data_set.read(value->flat<uint32>().data(), H5::PredType::NATIVE_UINT32, memory_space, data_space);
      } else if (H5Tequal(native_type, H5T_NATIVE_LONG)) {
        data_set.read(value->flat<int64>().data(), H5::PredType::NATIVE_LONG, memory_space, data_space);
      } else if (H5Tequal(native_type, H5T_NATIVE_FLOAT)) {
        data_set.read(value->flat<float>().data(), H5::PredType::NATIVE_FLOAT, memory_space, data_space);
      } else if (H5Tequal(native_type, H5T_NATIVE_DOUBLE)) {
        data_set.read(value->flat<double>().data(), H5::PredType::NATIVE_DOUBLE, memory_space, data_space);
      } else {
        return errors::Unimplemented("data type not supported yet: ", data_set.getTypeClass());
      }
    } catch(H5::FileIException e){
      return errors::InvalidArgument("unable to open dataset", e.getCDetailMsg());
    } 

    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("HDF5Indexable");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  std::unique_ptr<HDF5FileImage> file_image_;

  std::vector<DataType> dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
};

REGISTER_KERNEL_BUILDER(Name("HDF5IndexableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<HDF5Indexable>);
REGISTER_KERNEL_BUILDER(Name("HDF5IndexableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<HDF5Indexable>);
REGISTER_KERNEL_BUILDER(Name("HDF5IndexableRead").Device(DEVICE_CPU),
                        IOIndexableReadOp<HDF5Indexable>);
}  // namespace data
}  // namespace tensorflow
