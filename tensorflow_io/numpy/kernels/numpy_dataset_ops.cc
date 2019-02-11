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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace data {
namespace {

static const size_t kSyncMarkerSize = 16;
static const size_t kNumpyFileBufferSize = 1024 * 1024;

class NumpyFileReader {
 public:
  explicit NumpyFileReader(RandomAccessFile* file)
      : input_stream_(
            new io::BufferedInputStream(file, kNumpyFileBufferSize)) {}

  const std::vector<int64>& Shape() const {
    return shape_;
  }
  const std::string &Descr() const {
    return descr_;
  }
  size_t Count() const {
    if (shape_.size() == 0) {
      return 0;
    }
    size_t count = 1;
    for (size_t i = 0; i < shape_.size(); i++) {
      count *= shape_[i];
    }
    return count;
  }
  Status ReadNBytes(int64 bytes_to_read, string* result) {
    return input_stream_->ReadNBytes(bytes_to_read, result);
  }

  Status ReadHeader() {
    string magic;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(6, &magic));
    if (magic != "\x93NUMPY") {
      return errors::InvalidArgument(
          "numpy file header magic number invalid");
    }
    string version;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(2, &version));
    // TODO (yongtang): Support 2.0 which use 4 bytes for length.
    if (!(version[0] == 1 || version[1] == 0)) {
      return errors::InvalidArgument(
          "numpy file version only support 1.0: ", version[0], ".", version[1]);
    }
    string chunk;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(2, &chunk));
    int64 length = (uint64)(chunk[0]) + ((uint64)chunk[1] << 8);
    if ((magic.size() + version.size() + chunk.size() + length) % 16 != 0) {
      return errors::InvalidArgument(
          "numpy file header length is not aligned properly: ", length);
    }
    string dict;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(length, &dict));
    // {'descr': '<i8', 'fortran_order': False, 'shape': (4,), }\x20...\n
    if (dict.back() != '\n') {
      return errors::InvalidArgument(
          "numpy file header should end with '\\n'");
    }
    dict.pop_back();
    while (dict.back() == '\x20') {
        dict.pop_back();
    }
    Trim(dict);
    if (!(dict.front() == '{' && dict.back() == '}')) {
      return errors::InvalidArgument("numpy file header error: ", dict);
    }
    dict = dict.substr(1, dict.size()-2);
    Trim(dict);

    std::vector<std::pair<size_t, std::string> > positions;
    positions.push_back(std::pair<size_t, std::string>(dict.size(), ""));
    // find "'descr': ", "'fortran_order': ", "'shape': "
    std::vector<std::string> keys {"descr", "fortran_order", "shape"};
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
        Trim(value);
        if (value.back() == ',') {
            value.pop_back();
        }
        if (key == "descr") {
            // "'([<>|])([ifuc])(\\d+)'"
            if (!(value.front() == '\'' && value.back() == '\'')) {
                return errors::InvalidArgument("numpy file header error: ", dict);
            }
            value = value.substr(1, value.size()-2);
	    descr_ = value;
            if (!(value[0] == '<' || value[0] == '>' || value[0] == '|')) {
                return errors::InvalidArgument("numpy file header error: ", dict);
            }
            if (!(value[1] == 'i' || value[1] == 'f' || value[1] == 'u' || value[1] == 'c')) {
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
	    fortran_order_ = (value == "True");
        }
        if (key == "shape") {
            if (!(value.front() == '(' && value.back() == ')')) {
                return errors::InvalidArgument("numpy file header error: ", dict);
            }
            value = value.substr(1, value.size()-2);
            Trim(value);
	    shape_.clear();
            while (value.size() != 0) {
                size_t p = value.find(',');
                string number = value.substr(0, p);
                Trim(number);
                value = (p == std::string::npos) ? "" : value.substr(p + 1);
                Trim(value);
                int dim = std::stoul(number, &p);
                if (p != number.size() || dim == 0) {
                    return errors::InvalidArgument("numpy file header error: ", dict);
                }
		shape_.push_back(dim);
            }
        }
    }

    return Status::OK();
  }

  virtual ~NumpyFileReader() = default;

 private:
  void Trim(std::string& s) {
    const char *whitespace = " \t";
    size_t start = s.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        s = "";
        return;
    }
    size_t end = s.find_last_not_of(whitespace);
    s = s.substr(start, end - start + 1);
  }

  std::unique_ptr<io::InputStreamInterface> input_stream_;
  string descr_;
  bool fortran_order_;
  std::vector<int64> shape_;
  TF_DISALLOW_COPY_AND_ASSIGN(NumpyFileReader);
};
class NumpyFileDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit NumpyFileDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    // TODO (yongtang): remove restriction of output_types_?
    OP_REQUIRES(ctx, output_types_.size() == 1, errors::InvalidArgument("The number of elements in `output_types_` must be one."));
    for (const DataType& dt : output_types_) {
      OP_REQUIRES(ctx, dt == DT_INT32 || dt == DT_INT64,
                  errors::InvalidArgument(
                      "Each element of `output_types_` must be one of: "
                      "DT_INT32, DT_INT64"));
    }
  }
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    *output = new Dataset(ctx, filenames, output_types_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::vector<string>& filenames,
            const DataTypeVector& output_types)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(filenames),
          output_types_(output_types) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::NumpyFile")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{-1}});
      return *shapes;
    }

    string DebugString() const override {
      return "NumpyFileDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {filenames}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
	if (current_file_index_ == 0) {
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
	}
	if (current_file_index_ < dataset()->filenames_.size()) {
          const string& filename = dataset()->filenames_[current_file_index_];
          std::unique_ptr<RandomAccessFile> file_;
          std::unique_ptr<NumpyFileReader> reader_;
          TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(filename, &file_));
          reader_.reset(new NumpyFileReader(file_.get()));
          TF_RETURN_IF_ERROR(reader_->ReadHeader());
          TensorShape output_shape;
	  TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(reader_->Shape(), &output_shape));

	  std::string buffer;
	  auto output_type = dataset()->output_types_[0];
          Tensor value_tensor(ctx->allocator({}), output_type, output_shape);
	  switch (output_type)
	  {
          case DT_INT32: {
	      size_t length = reader_->Count() * sizeof(int32);
	      TF_RETURN_IF_ERROR(reader_->ReadNBytes(length, &buffer));
	      memcpy(value_tensor.flat<int32>().data(), buffer.data(), length);
	      break;
            }
          case DT_INT64: {
	      size_t length = reader_->Count() * sizeof(int64);
	      TF_RETURN_IF_ERROR(reader_->ReadNBytes(length, &buffer));
	      memcpy(value_tensor.flat<int64>().data(), buffer.data(), length);
	      break;
            }
	  }
          out_tensors->emplace_back(std::move(value_tensor));
          *end_of_sequence = false;
          ++current_file_index_;
	  return Status::OK();
	}
        *end_of_sequence = true;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal is currently not supported");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "RestoreInternal is currently not supported");
      }

     private:
      // Sets up NumpyFile streams to read at `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }
        return Status::OK();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
    };

    const std::vector<string> filenames_;
    const DataTypeVector output_types_;
  };
  DataTypeVector output_types_;
};

REGISTER_KERNEL_BUILDER(Name("NumpyFileDataset").Device(DEVICE_CPU),
                        NumpyFileDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
