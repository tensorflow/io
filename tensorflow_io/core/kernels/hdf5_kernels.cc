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

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

#include <H5Cpp.h>
#include <hdf5.h>
#include <hdf5_hl.h>

namespace tensorflow {
namespace data {
namespace {

class HDF5FileImage {
 public:
  HDF5FileImage(Env* env, const string& filename, const string& optional_memory)
      : filename_(filename), optional_memory_(optional_memory), file_(nullptr) {
    if (optional_memory.size() != 0) {
      file_image_ = H5LTopen_file_image(
          (void*)optional_memory_.data(), optional_memory_.size(),
          H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_DONT_RELEASE);
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
            file_image_ = H5LTopen_file_image(
                (void*)buffer_memory_.data(), buffer_memory_.size(),
                H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_DONT_RELEASE);
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

  H5::H5File* GetFile() const { return file_.get(); }

 private:
  string filename_;
  const string& optional_memory_;
  string buffer_memory_;
  std::unique_ptr<H5::H5File> file_;
  hid_t file_image_ = 0;
};

class HDF5Iterate {
 public:
  HDF5Iterate(haddr_t root) : parent_(root) { groups_[root] = ""; }
  ~HDF5Iterate() {}

  static herr_t Iterate(hid_t loc_id, const char* name, const H5L_info_t* info,
                        void* operator_data) {
    HDF5Iterate* p = (HDF5Iterate*)operator_data;

    H5O_info_t iteminfo;
    herr_t err = H5Oget_info_by_name(loc_id, name, &iteminfo, H5P_DEFAULT);

    switch (iteminfo.type) {
      case H5O_TYPE_GROUP:
        if (p->groups_.find(iteminfo.addr) == p->groups_.end()) {
          haddr_t parent = p->parent_;
          p->groups_[iteminfo.addr] = p->groups_[parent] + "/" + name;
          p->parent_ = iteminfo.addr;
          err = H5Literate_by_name(loc_id, name, H5_INDEX_NAME, H5_ITER_NATIVE,
                                   NULL, HDF5Iterate::Iterate, operator_data,
                                   H5P_DEFAULT);
          p->parent_ = parent;
        }
        break;
      case H5O_TYPE_DATASET: {
        string dataset = p->groups_[p->parent_] + "/" + name;
        p->datasets_.emplace_back(dataset);
      } break;
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
class HDF5ReadableResource : public ResourceBase {
 public:
  HDF5ReadableResource(Env* env)
      : env_(env), complex_names_(std::pair<string, string>("r", "i")) {}
  ~HDF5ReadableResource() {}

  Status Init(const string& input, std::vector<string>* components) {
    mutex_lock l(mu_);

    file_image_.reset(new HDF5FileImage(env_, input, ""));
    H5::H5File* file = file_image_->GetFile();
    if (file == nullptr) {
      return errors::InvalidArgument("unable to open hdf5 file: ", input);
    }

    H5O_info_t info;
    file->getObjinfo(info);
    HDF5Iterate data(info.addr);
    herr_t err = H5Literate(file->getId(), H5_INDEX_NAME, H5_ITER_NATIVE, NULL,
                            HDF5Iterate::Iterate, (void*)&data);

    for (size_t i = 0; i < data.datasets_.size(); i++) {
      ::tensorflow::DataType dtype;
      string dataset = data.datasets_[i];
      H5::DataSet data_set = file->openDataSet(dataset);

      H5::DataSpace data_space = data_set.getSpace();
      int rank = data_space.getSimpleExtentNdims();
      absl::InlinedVector<hsize_t, 4> dims(rank);
      data_space.getSimpleExtentDims(dims.data());

      H5::DataType data_type = data_set.getDataType();
      switch (data_type.getClass()) {
        case H5T_INTEGER:
          switch (data_type.getSize()) {
            case 1:
              dtype = static_cast<H5::IntType&>(data_type).getSign() ? DT_INT8
                                                                     : DT_UINT8;
              break;
            case 2:
              dtype = static_cast<H5::IntType&>(data_type).getSign()
                          ? DT_INT16
                          : DT_UINT16;
              break;
            case 4:
              dtype = static_cast<H5::IntType&>(data_type).getSign()
                          ? DT_INT32
                          : DT_UINT32;
              break;
            case 8:
              dtype = static_cast<H5::IntType&>(data_type).getSign()
                          ? DT_INT64
                          : DT_UINT64;
              break;
            default:
              return errors::InvalidArgument("unsupported data type size for ",
                                             dataset, ": ",
                                             data_type.getSize());
          }
          break;
        case H5T_FLOAT:
          switch (data_type.getSize()) {
            case 4:
              dtype = DT_FLOAT;
              break;
            case 8:
              dtype = DT_DOUBLE;
              break;
            default:
              return errors::InvalidArgument("unsupported data type size for ",
                                             dataset, ": ",
                                             data_type.getSize());
          }
          break;
        case H5T_STRING:
          dtype = DT_STRING;
          break;
        case H5T_VLEN:
          dtype = DT_STRING;
          break;
        case H5T_COMPOUND:
          if (static_cast<H5::CompType&>(data_type).getNmembers() != 2) {
            return errors::InvalidArgument(
                "unsupported compound members for ", dataset, ": ",
                static_cast<H5::CompType&>(data_type).getNmembers());
          }
          if (static_cast<H5::CompType&>(data_type).getMemberName(0) !=
                  complex_names_.first ||
              static_cast<H5::CompType&>(data_type).getMemberName(1) !=
                  complex_names_.second) {
            return errors::InvalidArgument(
                "unsupported compound member names for ", dataset, ": ",
                static_cast<H5::CompType&>(data_type).getMemberName(0), ", ",
                static_cast<H5::CompType&>(data_type).getMemberName(1));
          }
          if (static_cast<H5::CompType&>(data_type).getMemberDataType(0) !=
              static_cast<H5::CompType&>(data_type).getMemberDataType(1)) {
            return errors::InvalidArgument(
                "unsupported compound with different data type for ", dataset,
                ": ", static_cast<H5::CompType&>(data_type).getMemberClass(0),
                ", ", static_cast<H5::CompType&>(data_type).getMemberClass(1));
          }
          if (static_cast<H5::CompType&>(data_type).getMemberClass(0) !=
              H5T_FLOAT) {
            return errors::InvalidArgument(
                "unsupported compound with non-float data class for ", dataset,
                ": ", static_cast<H5::CompType&>(data_type).getMemberClass(0));
          }
          switch (static_cast<H5::CompType&>(data_type)
                      .getMemberDataType(0)
                      .getSize()) {
            case 4:
              dtype = DT_COMPLEX64;
              break;
            case 8:
              dtype = DT_COMPLEX128;
              break;
            default:
              return errors::InvalidArgument(
                  "unsupported data type size for compound", dataset, ": ",
                  static_cast<H5::CompType&>(data_type)
                      .getMemberDataType(0)
                      .getSize());
          }
          break;
        default:
          return errors::InvalidArgument("unsupported data class for ", dataset,
                                         ": ", data_type.getClass());
      }

      dtypes_.emplace_back(dtype);
      absl::InlinedVector<int64, 4> shape_dims(rank);
      for (int r = 0; r < rank; r++) {
        shape_dims[r] = dims[r];
      }
      shapes_.emplace_back(TensorShape(shape_dims));
    }

    columns_index_.reserve(data.datasets_.size());
    components->reserve(data.datasets_.size());
    for (size_t i = 0; i < data.datasets_.size(); i++) {
      columns_index_[data.datasets_[i]] = i;
      components->emplace_back(data.datasets_[i]);
    }

    return Status::OK();
  }
  Status Spec(const string& component, TensorShape* shape, DataType* dtype) {
    mutex_lock l(mu_);

    std::unordered_map<std::string, int64>::const_iterator lookup =
        columns_index_.find(component);
    if (lookup == columns_index_.end()) {
      return errors::InvalidArgument("dataset ", component, " not found");
    }
    const int64 column_index = lookup->second;
    *shape = shapes_[column_index];
    *dtype = dtypes_[column_index];
    return Status::OK();
  }
  Status Read(const string& component, const int64 start,
              const TensorShape& shape,
              std::function<Status(const TensorShape& shape, Tensor** value)>
                  allocate_func) {
    mutex_lock l(mu_);

    std::unordered_map<std::string, int64>::const_iterator lookup =
        columns_index_.find(component);
    if (lookup == columns_index_.end()) {
      return errors::InvalidArgument("dataset ", component, " not found");
    }
    const int64 column_index = lookup->second;

    Tensor* value;
    TF_RETURN_IF_ERROR(allocate_func(shape, &value));

    if (shape.dims() > 0) {
      if (shape.dim_size(0) == 0) {
        return Status::OK();
      }

      int64 element_start = start;

      if (element_start > shapes_[column_index].dim_size(0)) {
        return errors::InvalidArgument(
            "start ", element_start,
            " out of boundary: ", shapes_[column_index]);
      }
      int64 element_stop = element_start + shape.dim_size(0);
      if (element_stop > shapes_[column_index].dim_size(0)) {
        return errors::InvalidArgument(
            "start ", element_start, " and shape ", shape,
            " out of boundary: ", shapes_[column_index]);
      }
    }

    H5::H5File* file = file_image_->GetFile();
    try {
      H5::DataSet data_set = file->openDataSet(component);
      H5::DataType data_type = data_set.getDataType();
      H5::DataSpace data_space = data_set.getSpace();

      H5::DataSpace memory_space = H5::DataSpace::ALL;

      if (shape.dims() != 0) {
        int rank = data_space.getSimpleExtentNdims();
        absl::InlinedVector<hsize_t, 4> dims(rank);
        data_space.getSimpleExtentDims(dims.data());

        // Find the border of the dims start and dims
        absl::InlinedVector<hsize_t, 4> dims_start(dims.size(), 0);
        dims_start[0] = start;
        dims[0] = shape.dim_size(0);

        memory_space = H5::DataSpace(dims.size(), dims.data());

        data_space.selectHyperslab(H5S_SELECT_SET, dims.data(),
                                   dims_start.data());
      }
      switch (dtypes_[column_index]) {
        case DT_UINT8:
          data_set.read(value->flat<uint8>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_UINT16:
          data_set.read(value->flat<uint16>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_UINT32:
          data_set.read(value->flat<uint32>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_UINT64:
          data_set.read(value->flat<uint64>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_INT8:
          data_set.read(value->flat<int8>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_INT16:
          data_set.read(value->flat<int16>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_INT32:
          data_set.read(value->flat<int32>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_INT64:
          data_set.read(value->flat<int64>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_FLOAT:
          data_set.read(value->flat<float>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_DOUBLE:
          data_set.read(value->flat<double>().data(), data_type, memory_space,
                        data_space);
          break;
        case DT_COMPLEX64:
          data_set.read(value->flat<complex64>().data(), data_type,
                        memory_space, data_space);
          break;
        case DT_COMPLEX128:
          data_set.read(value->flat<complex128>().data(), data_type,
                        memory_space, data_space);
          break;
        case DT_STRING:
          switch (data_type.getClass()) {
            case H5T_STRING:
              if (data_set.getStrType().isVariableStr()) {
                int64 total = value->NumElements();
                std::unique_ptr<char*[]> buffer(new char*[total]);
                data_set.read(buffer.get(), data_set.getStrType(), memory_space,
                              data_space);
                for (int64 i = 0; i < value->NumElements(); i++) {
                  char* p = (char*)(buffer.get()[i]);
                  value->flat<string>()(i) = string(p);
                }
                H5::DataSet::vlenReclaim(buffer.get(), data_type, data_space);
              } else {
                int64 total = value->NumElements();
                std::unique_ptr<char[]> buffer(
                    new char[data_type.getSize() * total]);
                data_set.read(buffer.get(), data_type, memory_space,
                              data_space);

                switch (static_cast<H5::StrType&>(data_type).getStrpad()) {
                  case H5T_STR_NULLTERM:
                    for (int64 i = 0; i < value->NumElements(); i++) {
                      const char* p =
                          (const char*)(buffer.get() + data_type.getSize() * i);
                      size_t len = 0;
                      while (len < data_type.getSize() && p[len] != 0x00) {
                        len++;
                      }
                      value->flat<string>()(i) = string(p, len);
                    }
                    break;
                  case H5T_STR_NULLPAD:
                    for (int64 i = 0; i < value->NumElements(); i++) {
                      const char* p =
                          (const char*)(buffer.get() + data_type.getSize() * i);
                      size_t len = data_type.getSize();
                      while (len > 0 && p[len - 1] == 0x00) {
                        len--;
                      }
                      value->flat<string>()(i) = string(p, len);
                    }
                    break;
                  case H5T_STR_SPACEPAD:
                    return errors::InvalidArgument(
                        "string pad type not supported: ",
                        static_cast<H5::StrType&>(data_type).getStrpad());
                }
              }
              break;
            case H5T_VLEN: {
              int64 total = value->NumElements();
              std::unique_ptr<hvl_t[]> buffer(new hvl_t[total]);
              data_set.read(buffer.get(), data_type, memory_space, data_space);
              for (int64 i = 0; i < value->NumElements(); i++) {
                hvl_t* h = (hvl_t*)(buffer.get()) + i;
                value->flat<string>()(i) = string((const char*)(h->p), h->len);
              }
              H5::DataSet::vlenReclaim(buffer.get(), data_type, data_space);
            } break;
            default:
              return errors::Unimplemented(
                  "data type class for string not supported: ",
                  data_type.getClass());
          }
          break;
        default:
          return errors::Unimplemented("data type class not supported yet: ",
                                       data_type.getClass());
      }
    } catch (H5::FileIException e) {
      return errors::InvalidArgument("unable to open dataset",
                                     e.getCDetailMsg());
    }

    return Status::OK();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return "HDF5ReadableResource";
  }

 protected:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<HDF5FileImage> file_image_ GUARDED_BY(mu_);

  std::vector<DataType> dtypes_ GUARDED_BY(mu_);
  std::vector<TensorShape> shapes_ GUARDED_BY(mu_);
  std::unordered_map<string, int64> columns_index_ GUARDED_BY(mu_);

  std::pair<string, string> complex_names_ GUARDED_BY(mu_);
};

class HDF5ReadableInitOp : public ResourceOpKernel<HDF5ReadableResource> {
 public:
  explicit HDF5ReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<HDF5ReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<HDF5ReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<string>()();

    std::vector<string> components;
    OP_REQUIRES_OK(context, resource_->Init(input, &components));

    Tensor* components_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       1, TensorShape({static_cast<int64>(components.size())}),
                       &components_tensor));
    for (size_t i = 0; i < components.size(); i++) {
      components_tensor->flat<string>()(i) = components[i];
    }
  }
  Status CreateResource(HDF5ReadableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new HDF5ReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
class HDF5ReadableSpecOp : public OpKernel {
 public:
  explicit HDF5ReadableSpecOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    HDF5ReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* component_tensor;
    OP_REQUIRES_OK(context, context->input("component", &component_tensor));
    const string& component = component_tensor->scalar<string>()();

    TensorShape shape;
    DataType dtype;
    OP_REQUIRES_OK(context, resource->Spec(component, &shape, &dtype));

    Tensor* shape_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({shape.dims()}), &shape_tensor));
    for (int64 i = 0; i < shape.dims(); i++) {
      shape_tensor->flat<int64>()(i) = shape.dim_size(i);
    }

    Tensor* dtype_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &dtype_tensor));
    dtype_tensor->scalar<int64>()() = dtype;
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class HDF5ReadableReadOp : public OpKernel {
 public:
  explicit HDF5ReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    HDF5ReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    const int64 start = start_tensor->scalar<int64>()();

    const Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->input("shape", &shape_tensor));
    TensorShape shape(shape_tensor->flat<int64>());

    const Tensor* component_tensor;
    OP_REQUIRES_OK(context, context->input("component", &component_tensor));
    const string& component = component_tensor->scalar<string>()();

    OP_REQUIRES_OK(
        context,
        resource->Read(component, start, shape,
                       [&](const TensorShape& shape, Tensor** value) -> Status {
                         TF_RETURN_IF_ERROR(
                             context->allocate_output(0, shape, value));
                         return Status::OK();
                       }));
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>HDF5ReadableInit").Device(DEVICE_CPU),
                        HDF5ReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>HDF5ReadableSpec").Device(DEVICE_CPU),
                        HDF5ReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>HDF5ReadableRead").Device(DEVICE_CPU),
                        HDF5ReadableReadOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
