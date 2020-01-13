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
#include "tensorflow_io/core/kernels/io_stream.h"
#include "unzip.h"

namespace tensorflow {
namespace data {
namespace {

extern "C" {
struct zlib_fileopaque64_def {
  uint64 offset;
  uint64 length;
  ::tensorflow::RandomAccessFile* file;
};

voidpf filefunc_open64 OF((voidpf opaque, const void* filename, int mode)) {
  return opaque;
}
uLong filefunc_read OF((voidpf opaque, voidpf stream, void* buf, uLong size)) {
  struct zlib_fileopaque64_def* p = (struct zlib_fileopaque64_def*)opaque;
  StringPiece result;
  p->file->Read(p->offset, size, &result, (char*)buf);
  p->offset += result.size();
  return result.size();
}
uLong filefunc_write OF((voidpf opaque, voidpf stream, const void* buf,
                         uLong size)) {
  return -1;
}

ZPOS64_T filefunc_tell64 OF((voidpf opaque, voidpf stream)) {
  struct zlib_fileopaque64_def* p = (struct zlib_fileopaque64_def*)opaque;
  return p->offset;
}
long filefunc_seek64 OF((voidpf opaque, voidpf stream, ZPOS64_T offset,
                         int origin)) {
  struct zlib_fileopaque64_def* p = (struct zlib_fileopaque64_def*)opaque;
  int64 off = offset;
  switch (origin) {
    case ZLIB_FILEFUNC_SEEK_CUR:
      off = p->offset + offset;
      break;
    case ZLIB_FILEFUNC_SEEK_END:
      off = p->length - offset;
      break;
    case ZLIB_FILEFUNC_SEEK_SET:
      break;
    default:
      return -1;
  }
  if (off < 0 || off > p->length) {
    return -1;
  }
  p->offset = off;
  return 0;
}
int filefunc_close OF((voidpf opaque, voidpf stream)) { return 0; }
int filefunc_error OF((voidpf opaque, voidpf stream)) { return 0; }
}

void TrimNumpyHeader(std::string& s) {
  const char* whitespace = " \t";
  size_t start = s.find_first_not_of(whitespace);
  if (start == std::string::npos) {
    s = "";
    return;
  }
  size_t end = s.find_last_not_of(whitespace);
  s = s.substr(start, end - start + 1);
}

Status ParseNumpyHeader(io::InputStreamInterface* stream,
                        ::tensorflow::DataType* dtype,
                        std::vector<int64>* shape) {
  string descr;
  bool fortran_order = false;

  string magic;
  TF_RETURN_IF_ERROR(stream->ReadNBytes(6, &magic));
  if (magic != "\x93NUMPY") {
    return errors::InvalidArgument("numpy file header magic number invalid");
  }
  string version;
  TF_RETURN_IF_ERROR(stream->ReadNBytes(2, &version));
  // TODO (yongtang): Support 2.0 which use 4 bytes for length.
  if (!(version[0] == 1 || version[1] == 0)) {
    return errors::InvalidArgument(
        "numpy file version only support 1.0: ", version[0], ".", version[1]);
  }
  string chunk;
  TF_RETURN_IF_ERROR(stream->ReadNBytes(2, &chunk));
  int64 length = (uint64)(chunk[0]) + ((uint64)chunk[1] << 8);
  if ((magic.size() + version.size() + chunk.size() + length) % 16 != 0) {
    return errors::InvalidArgument(
        "numpy file header length is not aligned properly: ", length);
  }
  string dict;
  TF_RETURN_IF_ERROR(stream->ReadNBytes(length, &dict));
  // {'descr': '<i8', 'fortran_order': False, 'shape': (4,), }\x20...\n
  if (dict.back() != '\n') {
    return errors::InvalidArgument("numpy file header should end with '\\n'");
  }
  dict.pop_back();
  while (dict.back() == '\x20') {
    dict.pop_back();
  }
  TrimNumpyHeader(dict);
  if (!(dict.front() == '{' && dict.back() == '}')) {
    return errors::InvalidArgument("numpy file header error: ", dict);
  }
  dict = dict.substr(1, dict.size() - 2);
  TrimNumpyHeader(dict);

  std::vector<std::pair<size_t, std::string>> positions;
  positions.push_back(std::pair<size_t, std::string>(dict.size(), ""));
  // find "'descr': ", "'fortran_order': ", "'shape': "
  std::vector<std::string> keys{"descr", "fortran_order", "shape"};
  for (auto const& key : keys) {
    size_t p = dict.find("'" + key + "': ");
    if (p == std::string::npos) {
      return errors::InvalidArgument("numpy file header error: ", dict);
    }
    std::pair<size_t, std::string> position_pair(p, key);
    positions.push_back(std::pair<size_t, std::string>(p, key));
  }
  std::sort(positions.begin(), positions.end());
  for (int i = 0; i < positions.size() - 1; i++) {
    std::string key = positions[i].second;
    // "'<descr|fortran_order|shape>': "
    size_t value_offset = positions[i].first + positions[i].second.size() + 4;
    size_t value_length = positions[i + 1].first - value_offset;
    std::string value = dict.substr(value_offset, value_length);
    TrimNumpyHeader(value);
    if (value.back() == ',') {
      value.pop_back();
    }
    if (key == "descr") {
      // "'([<>|])([ifuc])(\\d+)'"
      if (!(value.front() == '\'' && value.back() == '\'')) {
        return errors::InvalidArgument("numpy file header error: ", dict);
      }
      value = value.substr(1, value.size() - 2);
      descr = value;
      if (!(value[0] == '<' || value[0] == '>' || value[0] == '|')) {
        return errors::InvalidArgument("numpy file header error: ", dict);
      }
      if (!(value[1] == 'i' || value[1] == 'f' || value[1] == 'u' ||
            value[1] == 'c')) {
        return errors::InvalidArgument("numpy file header error: ", dict);
      }
      value = value.substr(2);
      size_t p = 0;
      int n = std::stoul(value, &p);
      if (p != value.size() || n == 0) {
        return errors::InvalidArgument("numpy file header error: ", dict);
      }
    }
    if (key == "fortran_order") {
      if (value != "True" && value != "False") {
        return errors::InvalidArgument("numpy file header error: ", dict);
      }
      fortran_order = (value == "True");
    }
    if (key == "shape") {
      if (!(value.front() == '(' && value.back() == ')')) {
        return errors::InvalidArgument("numpy file header error: ", dict);
      }
      value = value.substr(1, value.size() - 2);
      TrimNumpyHeader(value);
      shape->clear();
      while (value.size() != 0) {
        size_t p = value.find(',');
        string number = value.substr(0, p);
        TrimNumpyHeader(number);
        value = (p == std::string::npos) ? "" : value.substr(p + 1);
        TrimNumpyHeader(value);
        int dim = std::stoul(number, &p);
        if (p != number.size() || dim == 0) {
          return errors::InvalidArgument("numpy file header error: ", dict);
        }
        shape->push_back(dim);
      }
    }
  }

  *dtype = ::tensorflow::DataType::DT_INVALID;
  if (!fortran_order) {
    if (descr == "|b1") {
      *dtype = ::tensorflow::DataType::DT_BOOL;
    } else if (descr == "|i1") {
      *dtype = ::tensorflow::DataType::DT_INT8;
    } else if (descr == "<i2") {
      *dtype = ::tensorflow::DataType::DT_INT16;
    } else if (descr == "<i4") {
      *dtype = ::tensorflow::DataType::DT_INT32;
    } else if (descr == "<i8") {
      *dtype = ::tensorflow::DataType::DT_INT64;
    } else if (descr == "|u1") {
      *dtype = ::tensorflow::DataType::DT_UINT8;
    } else if (descr == "<u2") {
      *dtype = ::tensorflow::DataType::DT_UINT16;
    } else if (descr == "<u4") {
      *dtype = ::tensorflow::DataType::DT_UINT32;
    } else if (descr == "<u8") {
      *dtype = ::tensorflow::DataType::DT_UINT64;
    } else if (descr == "<f4") {
      *dtype = ::tensorflow::DataType::DT_FLOAT;
    } else if (descr == "<f8") {
      *dtype = ::tensorflow::DataType::DT_DOUBLE;
    }
  }

  return Status::OK();
}

class ZipObjectInputStream : public io::InputStreamInterface {
 public:
  ZipObjectInputStream(unzFile uf) : uf_(uf) {}
  virtual ~ZipObjectInputStream() {}

  virtual Status ReadNBytes(int64 bytes_to_read, string* result) override {
    if (bytes_to_read < 0) {
      return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                     bytes_to_read);
    }

    result->clear();
    if (final_) {
      return errors::OutOfRange("EOF reached");
    }

    result->resize(bytes_to_read);
    int64 bytes_read = 0;
    while (bytes_read < bytes_to_read) {
      int err = unzReadCurrentFile(uf_, &((*result)[bytes_read]),
                                   bytes_to_read - bytes_read);
      if (err < 0) {
        result->resize(bytes_read);
        return errors::InvalidArgument(
            "error with zipfile in unzReadCurrentFile: ", err);
      }
      if (err == 0) {
        break;
      }
      bytes_read += err;
    }
    offset_ += bytes_read;
    result->resize(bytes_read);
    if (bytes_read < bytes_to_read) {
      return errors::OutOfRange("EOF reached");
    }
    return Status::OK();
  }

  virtual int64 Tell() const override { return offset_; }

  virtual Status Reset() override {
    return errors::Unimplemented("Reset zip object stream is not implemented");
  }

 private:
  unzFile uf_;
  int64 offset_ = 0;
  bool final_ = false;
};

Status NumpyInfo(const string& filename, const int64 size,
                 std::unique_ptr<tensorflow::RandomAccessFile>& file,
                 std::vector<string>* arrays,
                 std::vector<std::vector<int64>>* shapes,
                 std::vector<int64>* dtypes) {
  struct zlib_fileopaque64_def fileopaque;
  fileopaque.offset = 0;
  fileopaque.length = size;
  fileopaque.file = file.get();

  zlib_filefunc64_def filefunc;
  memset(&filefunc, 0x00, sizeof(zlib_filefunc64_def));
  filefunc.zopen64_file = filefunc_open64;
  filefunc.zread_file = filefunc_read;
  filefunc.zwrite_file = filefunc_write;
  filefunc.ztell64_file = filefunc_tell64;
  filefunc.zseek64_file = filefunc_seek64;
  filefunc.zclose_file = filefunc_close;
  filefunc.zerror_file = filefunc_error;
  filefunc.opaque = (voidpf)&fileopaque;

  unzFile uf = unzOpen2_64(filename.c_str(), &filefunc);
  if (uf == NULL) {
    // Not a zip file, try normal file
    io::RandomAccessInputStream stream(file.get());
    ::tensorflow::DataType dtype;
    std::vector<int64> shape;
    TF_RETURN_IF_ERROR(ParseNumpyHeader(&stream, &dtype, &shape));
    arrays->push_back("");
    shapes->push_back(shape);
    dtypes->push_back(dtype);
  } else {
    std::unique_ptr<unzFile, void (*)(unzFile*)> unzFile_scope_(
        &uf, [](unzFile* p) {
          if (p != nullptr) {
            unzClose(*p);
          }
        });
    unz_global_info64 gi;
    int err = unzGetGlobalInfo64(uf, &gi);
    if (err != UNZ_OK) {
      return errors::InvalidArgument("error with zipfile in unzGetGlobalInfo: ",
                                     err);
    }
    for (uLong i = 0; i < gi.number_entry; i++) {
      char filename_inzip[256];
      unz_file_info64 file_info;

      err = unzGetCurrentFileInfo64(uf, &file_info, filename_inzip,
                                    sizeof(filename_inzip), NULL, 0, NULL, 0);
      if (err != UNZ_OK) {
        errors::InvalidArgument("error with zipfile in unzGetCurrentFileInfo: ",
                                err);
      }

      size_t filename_inzip_len = strlen(filename_inzip);
      if (filename_inzip_len <= 4 ||
          memcmp(&filename_inzip[filename_inzip_len - 4], ".npy", 4)) {
        return errors::InvalidArgument("invalid name in zipfile: ",
                                       filename_inzip);
      }
      filename_inzip[filename_inzip_len - 4] = 0x00;

      err = unzOpenCurrentFile(uf);
      if (err != UNZ_OK) {
        return errors::InvalidArgument(
            "error with zipfile in unzOpenCurrentFile: ", err);
      }

      ZipObjectInputStream stream(uf);

      ::tensorflow::DataType dtype;
      std::vector<int64> shape;

      TF_RETURN_IF_ERROR(ParseNumpyHeader(&stream, &dtype, &shape));
      arrays->push_back(filename_inzip);
      shapes->push_back(shape);
      dtypes->push_back(dtype);

      if ((i + 1) < gi.number_entry) {
        err = unzGoToNextFile(uf);
        if (err != UNZ_OK) {
          return errors::InvalidArgument(
              "error with zipfile in unzGoToNextFile: ", err);
        }
      }
    }
  }

  return Status::OK();
}

class NumpyInfoOp : public OpKernel {
 public:
  explicit NumpyInfoOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    uint64 size;
    OP_REQUIRES_OK(context, env_->GetFileSize(filename, &size));

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    OP_REQUIRES_OK(context, env_->NewRandomAccessFile(filename, &file));

    std::vector<string> arrays;
    std::vector<std::vector<int64>> shapes;
    std::vector<int64> dtypes;

    OP_REQUIRES_OK(context,
                   NumpyInfo(filename, size, file, &arrays, &shapes, &dtypes));

    TensorShape output_shape = filename_tensor.shape();
    output_shape.AddDim(arrays.size());
    TensorShape shapes_shape = output_shape;
    size_t maxrank = 0;
    for (size_t i = 0; i < shapes.size(); i++) {
      maxrank = maxrank > shapes[i].size() ? maxrank : shapes[i].size();
    }
    shapes_shape.AddDim(maxrank);

    Tensor* arrays_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &arrays_tensor));

    Tensor* shapes_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, shapes_shape, &shapes_tensor));

    Tensor* dtypes_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, output_shape, &dtypes_tensor));

    for (size_t i = 0; i < arrays.size(); i++) {
      arrays_tensor->flat<string>()(i) = arrays[i];
      for (size_t j = 0; j < shapes[i].size(); j++) {
        shapes_tensor->flat<int64>()(i * maxrank + j) = shapes[i][j];
      }
      for (size_t j = shapes[i].size(); j < maxrank; j++) {
        shapes_tensor->flat<int64>()(i * maxrank + j) = -1;
      }
      dtypes_tensor->flat<int64>()(i) = dtypes[i];
    }
  }

 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class NumpySpecOp : public OpKernel {
 public:
  explicit NumpySpecOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    const Tensor& array_tensor = context->input(1);
    const string array = array_tensor.scalar<string>()();

    uint64 size;
    OP_REQUIRES_OK(context, env_->GetFileSize(filename, &size));

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    OP_REQUIRES_OK(context, env_->NewRandomAccessFile(filename, &file));

    std::vector<string> arrays;
    std::vector<std::vector<int64>> shapes;
    std::vector<int64> dtypes;

    OP_REQUIRES_OK(context,
                   NumpyInfo(filename, size, file, &arrays, &shapes, &dtypes));

    std::vector<int64> shape;
    int64 dtype = DT_INVALID;
    for (size_t i = 0; i < arrays.size(); i++) {
      if (arrays[i] == array) {
        shape = shapes[i];
        dtype = dtypes[i];
        break;
      }
    }
    OP_REQUIRES(context, (dtype != DT_INVALID),
                errors::InvalidArgument("unable to find array ", array, " in ",
                                        filename));

    TensorShape dtype_shape = filename_tensor.shape();
    TensorShape shape_shape = dtype_shape;
    shape_shape.AddDim(shape.size());

    Tensor* shape_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape_shape, &shape_tensor));

    Tensor* dtype_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, dtype_shape, &dtype_tensor));

    for (size_t i = 0; i < shape.size(); i++) {
      shape_tensor->flat<int64>()(i) = shape[i];
    }
    dtype_tensor->scalar<int64>()() = dtype;
  }

 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class NumpyReadOp : public OpKernel {
 public:
  explicit NumpyReadOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& address_tensor = context->input(0);
    const int64 address = address_tensor.scalar<int64>()();

    const Tensor& filename_tensor = context->input(1);
    const string& filename = filename_tensor.scalar<string>()();

    const Tensor& array_tensor = context->input(2);
    const string& array = array_tensor.scalar<string>()();

    const Tensor& shape_tensor = context->input(3);

    const Tensor& start_tensor = context->input(4);
    const int64 start = start_tensor.scalar<int64>()();

    const Tensor& stop_tensor = context->input(5);
    const int64 stop = stop_tensor.scalar<int64>()();

    if (address != 0) {
      size_t size = ::tensorflow::DataTypeSize(dtype_);
      std::vector<int64> shape;
      for (int64 i = 0; i < shape_tensor.NumElements(); i++) {
        shape.push_back(shape_tensor.flat<int64>()(i));
        size = size * shape_tensor.flat<int64>()(i);
      }
      std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(
          env_, filename, (const void*)(address), size));
      io::RandomAccessInputStream stream(file.get());
      OP_REQUIRES_OK(context, CopyNumpyToOutput(context, &stream, dtype_, shape,
                                                start, stop));
      return;
    }

    uint64 size;
    OP_REQUIRES_OK(context, env_->GetFileSize(filename, &size));

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    OP_REQUIRES_OK(context, env_->NewRandomAccessFile(filename, &file));

    struct zlib_fileopaque64_def fileopaque;
    fileopaque.offset = 0;
    fileopaque.length = size;
    fileopaque.file = file.get();

    zlib_filefunc64_def filefunc;
    memset(&filefunc, 0x00, sizeof(zlib_filefunc64_def));
    filefunc.zopen64_file = filefunc_open64;
    filefunc.zread_file = filefunc_read;
    filefunc.zwrite_file = filefunc_write;
    filefunc.ztell64_file = filefunc_tell64;
    filefunc.zseek64_file = filefunc_seek64;
    filefunc.zclose_file = filefunc_close;
    filefunc.zerror_file = filefunc_error;
    filefunc.opaque = (voidpf)&fileopaque;

    unzFile uf = unzOpen2_64(filename.c_str(), &filefunc);
    if (uf == NULL) {
      // Not a zip file, try normal file
      io::RandomAccessInputStream stream(file.get());
      ::tensorflow::DataType dtype;
      std::vector<int64> shape;
      OP_REQUIRES_OK(context, ParseNumpyHeader(&stream, &dtype, &shape));
      OP_REQUIRES_OK(context, CopyNumpyToOutput(context, &stream, dtype, shape,
                                                start, stop));
    } else {
      string entry = array + ".npy";
      int err = unzLocateFile(uf, entry.c_str(), 0);
      OP_REQUIRES(context, (err == UNZ_OK),
                  errors::InvalidArgument(
                      "error with zipfile in unzLocateFile: ", err));

      char filename_inzip[256];
      unz_file_info64 file_info;

      err = unzGetCurrentFileInfo64(uf, &file_info, filename_inzip,
                                    sizeof(filename_inzip), NULL, 0, NULL, 0);
      OP_REQUIRES(context, (err == UNZ_OK),
                  errors::InvalidArgument(
                      "error with zipfile in unzGetCurrentFileInfo: ", err));

      err = unzOpenCurrentFile(uf);
      OP_REQUIRES(context, (err == UNZ_OK),
                  errors::InvalidArgument(
                      "error with zipfile in unzOpenCurrentFile: ", err));

      ZipObjectInputStream stream(uf);

      ::tensorflow::DataType dtype;
      std::vector<int64> shape;
      OP_REQUIRES_OK(context, ParseNumpyHeader(&stream, &dtype, &shape));
      OP_REQUIRES_OK(context, CopyNumpyToOutput(context, &stream, dtype, shape,
                                                start, stop));

      unzClose(uf);
    }
  }

 protected:
  Status CopyNumpyToOutput(OpKernelContext* context,
                           io::InputStreamInterface* stream,
                           const ::tensorflow::DataType dtype,
                           const std::vector<int64>& shape, const int64 start,
                           const int64 stop) {
    int64 data_start = start;
    int64 data_stop = stop;
    if (data_start > shape[0]) {
      data_start = shape[0];
    }
    if (data_stop < 0) {
      data_stop = shape[0];
    }
    if (data_stop < data_start) {
      data_stop = data_start;
    }

    TensorShape output_shape;

    int64 slice = 1;

    output_shape.AddDim(data_stop - data_start);
    for (size_t i = 1; i < shape.size(); i++) {
      slice = slice * shape[i];

      output_shape.AddDim(shape[i]);
    }

#define PROCESS_TYPE(TYPE)                                                    \
  {                                                                           \
    int64 bytes_start =                                                       \
        data_start * slice * ::tensorflow::DataTypeSize(dtype);               \
    int64 bytes_stop = data_stop * slice * ::tensorflow::DataTypeSize(dtype); \
    TF_RETURN_IF_ERROR(stream->SkipNBytes(bytes_start));                      \
    Tensor* output_tensor;                                                    \
    TF_RETURN_IF_ERROR(                                                       \
        context->allocate_output(0, output_shape, &output_tensor));           \
    void* p = output_tensor->flat<TYPE>().data();                             \
    if (bytes_stop > bytes_start) {                                           \
      string buffer;                                                          \
      TF_RETURN_IF_ERROR(                                                     \
          stream->ReadNBytes(bytes_stop - bytes_start, &buffer));             \
      memcpy(p, buffer.data(), bytes_stop - bytes_start);                     \
    }                                                                         \
  }

    switch (dtype) {
      case ::tensorflow::DT_INT8:
        PROCESS_TYPE(int8);
        break;
      case ::tensorflow::DT_INT16:
        PROCESS_TYPE(int16);
        break;
      case ::tensorflow::DT_INT32:
        PROCESS_TYPE(int32);
        break;
      case ::tensorflow::DT_INT64:
        PROCESS_TYPE(int64);
        break;
      case ::tensorflow::DT_UINT8:
        PROCESS_TYPE(uint8);
        break;
      case ::tensorflow::DT_UINT16:
        PROCESS_TYPE(uint16);
        break;
      case ::tensorflow::DT_UINT32:
        PROCESS_TYPE(uint32);
        break;
      case ::tensorflow::DT_UINT64:
        PROCESS_TYPE(uint64);
        break;
      case ::tensorflow::DT_FLOAT:
        PROCESS_TYPE(float);
        break;
      case ::tensorflow::DT_DOUBLE:
        PROCESS_TYPE(double);
        break;
      default:
        return errors::InvalidArgument("unsupported type: ", dtype);
    }

    return Status::OK();
  }

 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  ::tensorflow::DataType dtype_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>NumpyInfo").Device(DEVICE_CPU), NumpyInfoOp);
REGISTER_KERNEL_BUILDER(Name("IO>NumpySpec").Device(DEVICE_CPU), NumpySpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>NumpyRead").Device(DEVICE_CPU), NumpyReadOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
