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

#include <archive.h>
#include <archive_entry.h>

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {
namespace data {
namespace {

class ArchiveInputStream : public io::InputStreamInterface {
 public:
  explicit ArchiveInputStream(RandomAccessFile* file, struct archive* archive)
    : file_(file), archive_(archive) {
  }
  Status ReadNBytes(int64 bytes_to_read, string* result) override {
    if (bytes_to_read < 0) {
      return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                     bytes_to_read);
    }
    result->clear();
    result->reserve(bytes_to_read);
    int64 bytes_read = 0;
    while (bytes_read < bytes_to_read) {
      ssize_t size = archive_read_data(archive_, &((*result)[bytes_read]), bytes_to_read - bytes_read);
      if (size == 0) {
        return errors::OutOfRange("EOF reached");
      }
      bytes_read += size;
      entry_offset_ += size;
    }
    return Status::OK();
  }
  virtual Status Reset() override {
    return errors::Unimplemented("not supported");
  }
  virtual int64 Tell() const override {
    return entry_offset_;
  }
  void ResetEntryOffset() {
    entry_offset_ = 0;
  }
  static ssize_t CallbackRead(struct archive *a, void *client_data, const void **buff) {
    class ArchiveInputStream *p = (class ArchiveInputStream *)client_data;
    StringPiece data(p->buffer_, sizeof(p->buffer_));
    Status s = p->file_->Read(p->pos_, sizeof(p->buffer_), &data, p->buffer_);
    if (!s.ok()) {
      if (!errors::IsOutOfRange(s)) {
        return -1;
      }
    }
    p->pos_ += data.size();
    *buff = p->buffer_;
    return data.size();
  }
  RandomAccessFile* file_;
  struct archive *archive_;
  char buffer_[4096];
  int64 pos_ = 0;
  int64 entry_offset_ = 0;
 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ArchiveInputStream);
};

class DataInput {
 public:
  DataInput()
   : filename_tensor_(DT_STRING, TensorShape({}))
   , entryname_tensor_(DT_STRING, TensorShape({})) {
    filename_tensor_.scalar<string>()() = "";
    entryname_tensor_.scalar<string>()() = "";
  }
  virtual Status ReadRecord(IteratorContext* ctx, io::InputStreamInterface& s, std::vector<Tensor>* out_tensors, bool* end_of_entry) const = 0;
  virtual Status FromStream(io::InputStreamInterface& s, const string& filename, const string& entryname) = 0;
  Status Initialize(Env* env, std::unique_ptr<ArchiveInputStream>& stream, std::unique_ptr<struct archive, void(*)(struct archive *)>& archive, std::unique_ptr<tensorflow::RandomAccessFile>& file) const {
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename_tensor_.scalar<string>()(), &file));

    archive.reset(archive_read_new());
    archive_read_support_compression_all(archive.get());
    archive_read_support_format_all(archive.get());

    stream.reset(new ArchiveInputStream(file.get(), archive.get()));
    if (archive_read_open(archive.get(), stream.get(), NULL, ArchiveInputStream::CallbackRead, NULL) != ARCHIVE_OK) {
      return errors::InvalidArgument("unable to open ", filename_tensor_.scalar<string>()(), ": ", archive_error_string(archive.get()));
    }
    struct archive_entry *entry;
    while (archive_read_next_header(archive.get(), &entry) == ARCHIVE_OK) {
      string entryname = archive_entry_pathname(entry);
      std::size_t found = entryname.find_last_of('/');
      if (found != string::npos) {
        entryname = entryname.substr(found + 1);
      }
      if (entryname_tensor_.scalar<string>()() == entryname) {
        stream->ResetEntryOffset();
        return Status::OK();
      }
    }
    return errors::InvalidArgument("unable to open ", entryname_tensor_.scalar<string>()(), " in ", filename_tensor_.scalar<string>()());
  }
  const string& filename() const {
    return filename_tensor_.scalar<string>()();
  }
  const string& entryname() const {
    return entryname_tensor_.scalar<string>()();
  }
 protected:
  Tensor filename_tensor_;
  Tensor entryname_tensor_;
};

template<typename T>
class DataInputOp: public OpKernel {
 public:
  explicit DataInputOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor* source_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source", &source_tensor));
    OP_REQUIRES(
        ctx, source_tensor->dims() <= 1,
        errors::InvalidArgument("`source` must be a scalar or a vector."));

    std::vector<string> source;
    source.reserve(source_tensor->NumElements());
    for (int i = 0; i < source_tensor->NumElements(); ++i) {
      source.push_back(source_tensor->flat<string>()(i));
    }

    std::vector<T> output(source.size());
    OP_REQUIRES_OK(ctx, Initialize(env_, source, &output));
    Tensor* output_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({static_cast<int64>(output.size())}), &output_tensor));
    for (int i = 0; i < output.size(); i++) {
      output_tensor->flat<Variant>()(i) = output[i];
    }
   }
 protected:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
 private:
  Status Initialize(Env *env, const std::vector<string>& source, std::vector<T>* output) {
    std::unordered_map<string, std::unordered_map<string, int>> key;
    for (int i = 0; i < source.size(); i++) {
      std::size_t found = source[i].find_first_of("|");
      string filename = source[i];
      string entryname = "";
      if (found != string::npos) {
        filename = source[i].substr(0, found);
	entryname = source[i].substr(found + 1);
      }
      if (key.find(filename) == key.end()) {
        key[filename] = std::unordered_map<string, int>();
      }
      key[filename][entryname] = i;
    }
    for (const auto& f : key) {
      const string& filename = f.first;
      std::unique_ptr<tensorflow::RandomAccessFile> file;
      TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

      std::unique_ptr<struct archive, void(*)(struct archive *)> archive(archive_read_new(), [](struct archive *a){ archive_read_free(a);});
      archive_read_support_compression_all(archive.get());
      archive_read_support_format_all(archive.get());

      ArchiveInputStream s(file.get(), archive.get());

      if (archive_read_open(archive.get(), &s, NULL, ArchiveInputStream::CallbackRead, NULL) != ARCHIVE_OK) {
        return errors::InvalidArgument("unable to open ", filename, ": ", archive_error_string(archive.get()));
      }
      struct archive_entry *entry;
      while (archive_read_next_header(archive.get(), &entry) == ARCHIVE_OK) {
        string entryname = archive_entry_pathname(entry);
        std::size_t found = entryname.find_last_of('/');
        if (found != string::npos) {
          entryname = entryname.substr(found + 1);
        }
	const auto& e = f.second.find(entryname);
	if (e != f.second.end()) {
          int index = e->second;
	  s.ResetEntryOffset();
	  T entry;
          TF_RETURN_IF_ERROR(entry.FromStream(s, filename, entryname));
	  (*output)[index] = std::move(entry);
	}
      }
    }
    return Status::OK();
  }
};
template<typename T>
class InputDatasetBase : public DatasetBase {
 public:
  InputDatasetBase(OpKernelContext* ctx, const std::vector<T>& input, const DataTypeVector& output_types, const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        ctx_(ctx),
        input_(input),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::unique_ptr<IteratorBase>(
        new Iterator({this, strings::StrCat(prefix, DebugString())}));
  }

  const DataTypeVector& output_dtypes() const override {
    return output_types_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return "InputDatasetBase::Dataset";
  }

  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** node) const override {
    return errors::Unimplemented(DebugString(), "::AsGraphDefInternal");
  }
 private:
  class Iterator : public DatasetIterator<InputDatasetBase<T>> {
   public:
    using tensorflow::data::DatasetIterator<InputDatasetBase<T>>::dataset;
    explicit Iterator(const typename tensorflow::data::DatasetIterator<InputDatasetBase<T>>::Params& params)
        : DatasetIterator<InputDatasetBase<T>>(params), stream_(nullptr), archive_(nullptr, [](struct archive *a){ archive_read_free(a);}), file_(nullptr){}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      do {
        if (stream_) {
          bool end_of_entry = false;
          TF_RETURN_IF_ERROR(dataset()->input_[current_input_index_].ReadRecord(ctx, (*stream_.get()), out_tensors, &end_of_entry));
          if (!end_of_entry) {
            *end_of_sequence = false;
            return Status::OK();
          }
          // We have reached the end of the current input, move next.
          ResetStreamsLocked();
          ++current_input_index_;
        }
        // Iteration ends when there are no more input to process.
        if (current_input_index_ == dataset()->input_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      } while (true);
    }

   private:
    // Sets up streams to read from `current_input_index_`.
    Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (current_input_index_ >= dataset()->input_.size()) {
        return errors::InvalidArgument(
            "current_input_index_:", current_input_index_,
            " >= input_.size():", dataset()->input_.size());
      }

      // Actually move on to next entry.
      TF_RETURN_IF_ERROR(dataset()->input_[current_input_index_].Initialize(env, stream_, archive_, file_));

      return Status::OK();
    }

    // Resets all streams.
    void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      stream_.reset(nullptr);
      archive_.reset(nullptr);
      file_.reset(nullptr);
    }

    mutex mu_;
    size_t current_input_index_ GUARDED_BY(mu_) = 0;
    std::unique_ptr<ArchiveInputStream> stream_ GUARDED_BY(mu_);
    std::unique_ptr<struct archive, void(*)(struct archive *)> archive_ GUARDED_BY(mu_);
    std::unique_ptr<tensorflow::RandomAccessFile> file_ GUARDED_BY(mu_);
  };
  OpKernelContext* ctx_;
 protected:
  std::vector<T> input_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

template<typename T>
class InputDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit InputDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(
        ctx, input_tensor->dims() <= 1,
        errors::InvalidArgument("`input` must be a scalar or a vector."));

    std::vector<T> input;
    input.reserve(input_tensor->NumElements());
    for (int i = 0; i < input_tensor->NumElements(); ++i) {
      input.push_back(*(input_tensor->flat<Variant>()(i).get<T>()));
    }
    *output = new InputDatasetBase<T>(ctx, input, output_types_, output_shapes_);
  }
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class CIFAR10Input : public DataInput {
 public:
  CIFAR10Input() : DataInput(), size_tensor_(DT_INT64, TensorShape({})) {
    size_tensor_.scalar<int64>()() = 0;
  }
  void Encode(VariantTensorData* data) const {
    data->tensors_ = {filename_tensor_, entryname_tensor_, size_tensor_};
  }
  bool Decode(const VariantTensorData& data) {
    filename_tensor_ = data.tensors(0);
    entryname_tensor_ = data.tensors(1);
    size_tensor_ = data.tensors(2);
    return true;
  }
  Status ReadRecord(IteratorContext* ctx, io::InputStreamInterface& s, std::vector<Tensor>* out_tensors, bool* end_of_entry) const {
    string buffer;
    Status status = s.ReadNBytes(3073, &buffer);
    if (!status.ok()) {
      if (status != errors::OutOfRange("EOF reached")) {
	return status;
      }
      *end_of_entry = true;
      return Status::OK();
    }

    Tensor label_tensor(ctx->allocator({}), DT_UINT8, {});
    label_tensor.scalar<uint8>()() = *((uint8 *)(&(buffer.data()[0])));
    out_tensors->emplace_back(std::move(label_tensor));

    Tensor value_tensor(ctx->allocator({}), DT_UINT8, {3, 32, 32});
    // TODO(yongtang): Proess data here?
    //Eigen::array<int, 3> p({2, 0, 1});
    //auto x = typename TTypes<uint8, 3>::ConstTensor(reinterpret_cast<const uint8*>(&(value.data()[1])), {3, 32, 32});
    //value_tensor.tensor<uint8, 3>().device(dataset()->ctx_->eigen_device<Eigen::ThreadPoolDevice>()) = x.shuffle(p);
    memcpy(value_tensor.flat<uint8>().data(), &(buffer.data()[1]), 3072);
    out_tensors->emplace_back(std::move(value_tensor));

    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface& s, const string& filename, const string& entryname) {
    int64 size = 0;
    Status status = s.SkipNBytes(3073);
    while (status.ok()) {
      size += 1;
      status = s.SkipNBytes(3073);
    }
    if (status != errors::OutOfRange("EOF reached")) {
      return status;
    }
    filename_tensor_.scalar<string>()() = filename;
    entryname_tensor_.scalar<string>()() = entryname;
    size_tensor_.scalar<int64>()() = size;
    return Status::OK();
  }
  int64 size() const {
    return size_tensor_.scalar<int64>()();
  }
 private:
  Tensor size_tensor_;
};
class CIFAR100Input : public DataInput {
 public:
  CIFAR100Input() : DataInput(), size_tensor_(DT_INT64, TensorShape({})) {
    size_tensor_.scalar<int64>()() = 0;
  }
  void Encode(VariantTensorData* data) const {
    data->tensors_ = {filename_tensor_, entryname_tensor_, size_tensor_};
  }
  bool Decode(const VariantTensorData& data) {
    filename_tensor_ = data.tensors(0);
    entryname_tensor_ = data.tensors(1);
    size_tensor_ = data.tensors(2);
    return true;
  }
  Status ReadRecord(IteratorContext* ctx, io::InputStreamInterface& s, std::vector<Tensor>* out_tensors, bool* end_of_entry) const {
    string buffer;
    Status status = s.ReadNBytes(3074, &buffer);
    if (!status.ok()) {
      if (status != errors::OutOfRange("EOF reached")) {
       return status;
      }
      *end_of_entry = true;
      return Status::OK();
    }

    Tensor coarse_tensor(ctx->allocator({}), DT_UINT8, {});
    coarse_tensor.scalar<uint8>()() = *((uint8 *)(&(buffer.data()[0])));
    out_tensors->emplace_back(std::move(coarse_tensor));

    Tensor fine_tensor(ctx->allocator({}), DT_UINT8, {});
    fine_tensor.scalar<uint8>()() = *((uint8 *)(&(buffer.data()[1])));
    out_tensors->emplace_back(std::move(fine_tensor));

    Tensor value_tensor(ctx->allocator({}), DT_UINT8, {3, 32, 32});
    // TODO(yongtang): Proess data here?
    //Eigen::array<int, 3> p({2, 0, 1});
    //auto x = typename TTypes<uint8, 3>::ConstTensor(reinterpret_cast<const uint8*>(&(value.data()[1])), {3, 32, 32});
    //value_tensor.tensor<uint8, 3>().device(dataset()->ctx_->eigen_device<Eigen::ThreadPoolDevice>()) = x.shuffle(p);
    memcpy(value_tensor.flat<uint8>().data(), &(buffer.data()[2]), 3072);
    out_tensors->emplace_back(std::move(value_tensor));

    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface& s, const string& filename, const string& entryname) {
    int64 size = 0;
    Status status = s.SkipNBytes(3074);
    while (status.ok()) {
      size += 1;
      status = s.SkipNBytes(3074);
    }
    if (status != errors::OutOfRange("EOF reached")) {
      return status;
    }
    filename_tensor_.scalar<string>()() = filename;
    entryname_tensor_.scalar<string>()() = entryname;
    size_tensor_.scalar<int64>()() = size;
    return Status::OK();
  }
  int64 size() const {
    return size_tensor_.scalar<int64>()();
  }
 private:
  Tensor size_tensor_;
};
 
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CIFAR10Input, "tensorflow::CIFAR10Input");
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CIFAR100Input, "tensorflow::CIFAR100Input");

REGISTER_KERNEL_BUILDER(Name("CIFAR10Input").Device(DEVICE_CPU),
                        DataInputOp<CIFAR10Input>);
REGISTER_KERNEL_BUILDER(Name("CIFAR100Input").Device(DEVICE_CPU),
                        DataInputOp<CIFAR100Input>);
REGISTER_KERNEL_BUILDER(Name("CIFAR10Dataset").Device(DEVICE_CPU),
                        InputDatasetOp<CIFAR10Input>);
REGISTER_KERNEL_BUILDER(Name("CIFAR100Dataset").Device(DEVICE_CPU),
                        InputDatasetOp<CIFAR100Input>);
}  // namespace
}  // namespace data
}  // namespace tensorflow
