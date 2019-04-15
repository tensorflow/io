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
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {
namespace data {

class ArchiveInputStream : public io::InputStreamInterface {
 public:
  explicit ArchiveInputStream(RandomAccessFile* file, struct archive* archive)
    : file_(file), archive_(archive) {
  }
  static Status SetupFilters(struct archive *archive, const std::vector<string>& filters) {
    for (const auto& filter : filters) {
      if (filter == "none") {
        archive_read_support_filter_none(archive);
        archive_read_support_format_raw(archive);
	continue;
      }
      if (filter == "gz") {
        archive_read_support_filter_gzip(archive);
        archive_read_support_format_raw(archive);
	continue;
      }
      string name = filter;
      std::size_t found = filter.find_first_of(':');
      if (found != string::npos) {
        name = filter.substr(0, found);
      }
      if (name == "tar.gz") {
        archive_read_support_filter_gzip(archive);
        archive_read_support_format_tar(archive);
      }
    }
    return Status::OK();
  }
  static bool MatchFilters(struct archive *archive, const string& entryname, const std::vector<string>& filters, string* filtername) {
    string archive_format(archive_format_name(archive));
    std::vector<string> archive_filter(archive_filter_count(archive));
    for (int i = 0; i < archive_filter.size(); i++) {
      archive_filter[i] = archive_filter_name(archive, i);
    }
    for (const auto& filter : filters) {
      if (filter == "none") {
        if (archive_format == "raw" && archive_filter.size() == 1 && archive_filter[0] == "none") {
          *filtername = "none";
          return true;
	}
      }
      if (filter == "gz") {
        if (archive_format == "raw" && archive_filter.size() > 0 && archive_filter[0] == "gzip") {
          *filtername = "gz";
          return true;
	}
      }
      string name = filter;
      string fname = "";
      std::size_t found = filter.find_first_of(':');
      if (found != string::npos) {
        name = filter.substr(0, found);
	fname = filter.substr(found + 1);
      }
      string ename = entryname;
      found = entryname.find_last_of('/');
      if (found != string::npos) {
        ename = entryname.substr(found + 1);
      }
      if (name == "tar.gz") {
        if (archive_format == "GNU tar format" && archive_filter.size() > 0 && archive_filter[0] == "gzip" && fname == ename) {
          *filtername = "tar.gz";
          return true;
	}
      }
    }
    return false;
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

template<typename T>
class DataInput {
 public:
  DataInput() {}
  virtual ~DataInput() {}
  virtual Status FromStream(io::InputStreamInterface& s) = 0;
  virtual Status ReadRecord(io::InputStreamInterface& s, IteratorContext* ctx, std::unique_ptr<T>& state, int64* returned, std::vector<Tensor>* out_tensors) const = 0;
  virtual void EncodeAttributes(VariantTensorData* data) const = 0;
  virtual bool DecodeAttributes(const VariantTensorData& data) = 0;

  Status ReadInputStream(io::InputStreamInterface& s, int64 chunk, int64 count, string* buffer, int64* returned) const {
    int64 offset = s.Tell();
    int64 bytes_to_read = count * chunk;
    Status status = (buffer == nullptr) ? s.SkipNBytes(bytes_to_read) : s.ReadNBytes(bytes_to_read, buffer);
    if (!(status.ok() || status == errors::OutOfRange("EOF reached"))) {
      return status;
    }
    int64 bytes_read = s.Tell() - offset;
    if (bytes_read % chunk != 0) {
      return errors::DataLoss("corrupted data, expected multiple of ", chunk, ", received ", bytes_read);
    }
    *returned = bytes_read / chunk;
    return Status::OK();
  }
  Status FromInputStream(io::InputStreamInterface& s, const string& filename, const string& entryname, const string& filtername) {
    filename_ = filename;
    entryname_ = entryname;
    filtername_ = filtername;
    return FromStream(s);
  }
  void Encode(VariantTensorData* data) const {
    data->tensors_ = {Tensor(DT_STRING, TensorShape({})), Tensor(DT_STRING, TensorShape({})), Tensor(DT_STRING, TensorShape({}))};
    data->tensors_[0].scalar<string>()() = filename_;
    data->tensors_[1].scalar<string>()() = entryname_;
    data->tensors_[2].scalar<string>()() = filtername_;

    EncodeAttributes(data);
  }
  bool Decode(const VariantTensorData& data) {
    filename_ = data.tensors(0).scalar<string>()();
    entryname_ = data.tensors(1).scalar<string>()();
    filtername_ = data.tensors(2).scalar<string>()();

    return DecodeAttributes(data);
  }
  const string& filename() const {
    return filename_;
  }
  const string& entryname() const {
    return entryname_;
  }
  const string& filtername() const {
    return filtername_;
  }
 protected:
  string filename_;
  string entryname_;
  string filtername_;
};

template<typename T>
class DataInputOp: public OpKernel {
 public:
  explicit DataInputOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("filters", &filters_));
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

    std::vector<T> output;

    for (const auto& filename: source) {
      std::unique_ptr<tensorflow::RandomAccessFile> file;
      OP_REQUIRES_OK(ctx, env_->NewRandomAccessFile(filename, &file));
      if (filters_.size() == 0) {
        // No filter means only a file stream.
        io::RandomAccessInputStream s(file.get());
        T entry;
        OP_REQUIRES_OK(ctx, entry.FromInputStream(s, filename, string(""), string("")));
        output.emplace_back(std::move(entry));
        continue;
      }

      std::unique_ptr<struct archive, void(*)(struct archive *)> archive(archive_read_new(), [](struct archive *a){ archive_read_free(a);});
      OP_REQUIRES_OK(ctx, ArchiveInputStream::SetupFilters(archive.get(), filters_));

      ArchiveInputStream s(file.get(), archive.get());

      OP_REQUIRES(
          ctx, (archive_read_open(archive.get(), &s, NULL, ArchiveInputStream::CallbackRead, NULL) == ARCHIVE_OK), 
          errors::InvalidArgument("unable to open datainput for ", filename, ": ", archive_error_string(archive.get())));

      size_t index = output.size();

      struct archive_entry *entry;
      while (archive_read_next_header(archive.get(), &entry) == ARCHIVE_OK) {
        string entryname = archive_entry_pathname(entry);
	string filtername;
        if (ArchiveInputStream::MatchFilters(archive.get(), entryname, filters_, &filtername)) {
          s.ResetEntryOffset();
	  T entry;
          OP_REQUIRES_OK(ctx, entry.FromInputStream(s, filename, entryname, filtername));
          output.emplace_back(std::move(entry));
	}
      }
      std::sort(output.begin() + index, output.end(), [](const T& a, const T& b) {
          return a.entryname() < b.entryname();
      });
    }

    Tensor* output_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({static_cast<int64>(output.size())}), &output_tensor));
    for (int i = 0; i < output.size(); i++) {
      output_tensor->flat<Variant>()(i) = output[i];
    }
   }
 protected:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::vector<string> filters_ GUARDED_BY(mu_);
};
template<typename InputType, typename StateType>
class InputDatasetBase : public DatasetBase {
 public:
  InputDatasetBase(OpKernelContext* ctx, const std::vector<InputType>& input, const DataTypeVector& output_types, const std::vector<PartialTensorShape>& output_shapes)
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
    Node* input_node;
    Tensor input_tensor(DT_STRING, TensorShape({static_cast<int64>(input_.size())}));
    // GraphDefInternal has some trouble with Variant so use serialized string.
    for (size_t i = 0; i < input_.size(); i++) {
      string message;
      VariantTensorData serialized_data_f;
      VariantTensorDataProto serialized_proto_f;
      input_[i].Encode(&serialized_data_f);
      serialized_data_f.ToProto(&serialized_proto_f);
      EncodeVariant(serialized_proto_f, &message);
      input_tensor.flat<string>()(i) = message;
    }
    TF_RETURN_IF_ERROR(b->AddTensor(input_tensor, &input_node));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_node }, node));
    return Status::OK();
  }
 private:
  class Iterator : public DatasetIterator<InputDatasetBase<InputType, StateType>> {
   public:
    using tensorflow::data::DatasetIterator<InputDatasetBase<InputType, StateType>>::dataset;
    explicit Iterator(const typename tensorflow::data::DatasetIterator<InputDatasetBase<InputType, StateType>>::Params& params)
        : DatasetIterator<InputDatasetBase<InputType, StateType>>(params), stream_(nullptr), archive_(nullptr, [](struct archive *a){ archive_read_free(a);}), file_(nullptr){}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      do {
        if (stream_) {
	  int64 count = 1;
          int64 returned = 0;
          TF_RETURN_IF_ERROR(dataset()->input_[current_input_index_].ReadRecord((*stream_.get()), ctx, current_input_state_, &returned, out_tensors));
          if (returned == count) {
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
      const string& filename = dataset()->input_[current_input_index_].filename();
      const string& entryname = dataset()->input_[current_input_index_].entryname();
      const string& filtername = dataset()->input_[current_input_index_].filtername();

      current_input_state_.reset(nullptr);

      TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file_));
      if (filtername.size() == 0) {
        // No filter means only a file stream.
        stream_.reset(new io::RandomAccessInputStream(file_.get()));
        return Status::OK();
      }
      archive_.reset(archive_read_new());

      std::vector<string> filters(1, filtername);
      TF_RETURN_IF_ERROR(ArchiveInputStream::SetupFilters(archive_.get(), filters));

      stream_.reset(new ArchiveInputStream(file_.get(), archive_.get()));
      if (archive_read_open(archive_.get(), stream_.get(), NULL, ArchiveInputStream::CallbackRead, NULL) != ARCHIVE_OK) {
        return errors::InvalidArgument("unable to open dataset for ", filename, ": ", archive_error_string(archive_.get()));
      }

      struct archive_entry *entry;
      while (archive_read_next_header(archive_.get(), &entry) == ARCHIVE_OK) {
        if (entryname == archive_entry_pathname(entry)) {
          static_cast<ArchiveInputStream *>(stream_.get())->ResetEntryOffset();

          return Status::OK();
	}
      }
      return errors::InvalidArgument("unable to open ", filename, "|", entryname, ": ", archive_error_string(archive_.get()));
    }

    // Resets all streams.
    void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      current_input_state_.reset(nullptr);
      stream_.reset(nullptr);
      archive_.reset(nullptr);
      file_.reset(nullptr);
    }

    mutex mu_;
    size_t current_input_index_ GUARDED_BY(mu_) = 0;
    std::unique_ptr<StateType> current_input_state_ GUARDED_BY(mu_);
    std::unique_ptr<io::InputStreamInterface> stream_ GUARDED_BY(mu_);
    std::unique_ptr<struct archive, void(*)(struct archive *)> archive_ GUARDED_BY(mu_);
    std::unique_ptr<tensorflow::RandomAccessFile> file_ GUARDED_BY(mu_);
  };
  OpKernelContext* ctx_;
 protected:
  std::vector<InputType> input_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

template<typename InputType, typename StateType>
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
        ctx, (input_tensor->dtype() == DT_VARIANT || input_tensor->dtype() == DT_STRING),
        errors::InvalidArgument("`input` must be a variant or string, received ", input_tensor->dtype()));
    OP_REQUIRES(
        ctx, input_tensor->dims() <= 1,
        errors::InvalidArgument("`input` must be a scalar or a vector, dim = ", input_tensor->dims()));
    std::vector<InputType> input;
    input.reserve(input_tensor->NumElements());
    if (input_tensor->dtype() == DT_VARIANT) {
      for (int i = 0; i < input_tensor->NumElements(); ++i) {
        input.push_back(*(input_tensor->flat<Variant>()(i).get<InputType>()));
      }
    } else {
      for (int i = 0; i < input_tensor->NumElements(); ++i) {
        string message = input_tensor->flat<string>()(i);
        VariantTensorDataProto serialized_proto_f;
        VariantTensorData serialized_data_f;
        DecodeVariant(&message, &serialized_proto_f);
        serialized_data_f.FromProto(serialized_proto_f);
        InputType entry;
        entry.Decode(serialized_data_f);
        input.emplace_back(entry);
      }
    }
    *output = new InputDatasetBase<InputType, StateType>(ctx, input, output_types_, output_shapes_);
  }
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};
}  // namespace data
}  // namespace tensorflow
