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
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {
namespace data {

class SizedRandomAccessInputStreamInterface : public io::InputStreamInterface {
public:
  virtual Status GetFileSize(uint64* file_size) = 0;
  virtual Status Read(uint64 offset, size_t n, StringPiece* result,
                      char* scratch) const = 0;
};

class SizedRandomAccessFileStream : public SizedRandomAccessInputStreamInterface {
public:
  explicit SizedRandomAccessFileStream(RandomAccessFile *file, uint64 size)
    : file_(file)
    , size_(size)
    , file_stream_(file) {}
  Status GetFileSize(uint64* file_size) override {
    *file_size = size_;
    return Status::OK();
  }
  Status Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const override {
    return file_->Read(offset, n , result, scratch);
  }
  Status ReadNBytes(int64 bytes_to_read, string* result) override {
    return file_stream_.ReadNBytes(bytes_to_read, result);
  }
  int64 Tell() const override {
    return file_stream_.Tell();
  }
  Status Reset() override {
    return file_stream_.Reset();
  }
private:
  RandomAccessFile *file_;
  uint64 size_;
  io::RandomAccessInputStream file_stream_;
  string buffer_;
};

class SizedRandomAccessBufferedStream : public SizedRandomAccessInputStreamInterface {
public:
  explicit SizedRandomAccessBufferedStream(io::InputStreamInterface* s)
    : input_stream_(s) { }
  Status GetFileSize(uint64* file_size) override {
    // TODO: This is not necessary the best format as it needs
    // two pass to get the buffer. Could be enhanced later.
    if (size_ >= 0) {
      *file_size = size_;
      return Status::OK();
    }
    std::vector<string> buffer;
    do {
      string chunk;
      Status status = input_stream_->ReadNBytes(4096, &chunk);
      if (!(status.ok() || errors::IsOutOfRange(status))) {
        return status;
      }
      if (chunk.size() > 0) {
        buffer.emplace_back(std::move(chunk));
      }
      if (!status.ok()) {
        break;
      }
    } while (true);
    size_ = 0;
    for (size_t i = 0; i < buffer.size(); i++) {
        size_ += buffer[i].size();
    }
    buffer_.clear();
    buffer_.reserve(size_);
    for (size_t i = 0; i < buffer.size(); i++) {
        buffer_.append(buffer[i]);
    }
    buffer.clear();

    *file_size = size_;
    return  Status::OK();
  }
  Status Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const override {
    Status status = Status::OK();
    if (offset + n > size_) {
      status = errors::OutOfRange("EOF reached: ", result->size(), " bytes read, ", n, " requested");
      n = size_ - offset;
    }
    memcpy(scratch, &buffer_.data()[offset], n);
    *result = StringPiece(scratch, n);
    return status;
  }
  Status ReadNBytes(int64 bytes_to_read, string* result) override {
    return input_stream_->ReadNBytes(bytes_to_read, result);
  }
  int64 Tell() const override {
    return input_stream_->Tell();
  }
  Status Reset() override {
    return input_stream_->Reset();
  }
private:
  io::InputStreamInterface* input_stream_;
  string buffer_;
  int64 size_ = -1;
};

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

// Note: Forward declaration for friend class.
template<typename T> class FileInput;
template<typename T> class StreamInput;

template<typename T>
class DataInput {
 public:
  DataInput() {}
  virtual ~DataInput() {}
 protected:
  virtual void EncodeAttributes(VariantTensorData* data) const = 0;
  virtual bool DecodeAttributes(const VariantTensorData& data) = 0;
  virtual Status ReadReferenceRecord(void* s, IteratorContext* ctx, std::unique_ptr<T>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const = 0;
  Status ReadReferenceBatchRecord(void* s, IteratorContext* ctx, std::unique_ptr<T>& state, int64 batch, int64 count, int64* returned, std::vector<Tensor>* out_tensors) const {
    int64 record_read = 0;
    int64 record_to_read = count - (*returned);
    std::vector<Tensor> chunk_tensors;
    TF_RETURN_IF_ERROR(ReadReferenceRecord(s, ctx, state, record_to_read, &record_read, &chunk_tensors));
    if (record_read > 0) {
      if (out_tensors->size() == 0) {
        // Replace out_tensors with chunk_tensors
        out_tensors->reserve(chunk_tensors.size());
        // batch == 0 could only read at most one record
        // so it only happens here.
        if (batch == 0) {
          for (size_t i = 0; i < chunk_tensors.size(); i++) {
            TensorShape shape = chunk_tensors[i].shape();
            shape.RemoveDim(0);
            Tensor value_tensor(ctx->allocator({}), chunk_tensors[i].dtype(), shape);
            value_tensor.CopyFrom(chunk_tensors[i], shape);
            out_tensors->emplace_back(std::move(value_tensor));
          }
        } else {
          for (size_t i = 0; i < chunk_tensors.size(); i++) {
            out_tensors->emplace_back(std::move(chunk_tensors[i]));
          }
        }
      } else {
        // Append out_tensors with chunk_tensors
        for (size_t i = 0; i < out_tensors->size(); i++) {
          TensorShape shape = (*out_tensors)[i].shape();
          shape.set_dim(0, shape.dim_size(0) + record_read);
          Tensor value_tensor(ctx->allocator({}), (*out_tensors)[i].dtype(), shape);
          TensorShape element_shape = shape;
          element_shape.RemoveDim(0);
          Tensor element(ctx->allocator({}), (*out_tensors)[i].dtype(), element_shape);
          for (size_t index = 0; index < (*out_tensors)[i].shape().dim_size(0); index++) {
            TF_RETURN_IF_ERROR(batch_util::CopySliceToElement((*out_tensors)[i], &element, index));
            TF_RETURN_IF_ERROR(batch_util::CopyElementToSlice(element, &value_tensor, index));
          }
          for (size_t index = 0; index < record_read; index++) {
            TF_RETURN_IF_ERROR(batch_util::CopySliceToElement(chunk_tensors[i], &element, index));
            TF_RETURN_IF_ERROR(batch_util::CopyElementToSlice(element, &value_tensor, (*out_tensors)[i].shape().dim_size(0) + index));
          }
          (*out_tensors)[i] = std::move(value_tensor);
        }
      }
      (*returned) += record_read;
    }
    return Status::OK();
  }
  friend class FileInput<T>;
  friend class StreamInput<T>;
};


template<typename T>
class FileInput : public DataInput<T> {
 public:

  Status FromInputStream(io::InputStreamInterface* s, const string& filename,
    const string& entryname, const string& filtername, const string& schema,
    const std::vector<string>& columns) {

    filename_ = filename;
    entryname_ = entryname;
    filtername_ = filtername;
    schema_ = schema;
    columns_ = columns;
    return FromStream(s);
  }

  Status ReadBatchRecord(io::InputStreamInterface* s, IteratorContext* ctx,
    std::unique_ptr<T>& state, int64 batch, int64 count, int64* returned,
    std::vector<Tensor>* out_tensors) const {

    return (static_cast<const DataInput<T> *>(this))->ReadReferenceBatchRecord(
      static_cast<void *>(s), ctx, state, batch, count, returned, out_tensors);
  }
  void Encode(VariantTensorData* data) const {
    data->tensors_ = {
        Tensor(DT_STRING, TensorShape({})),
        Tensor(DT_STRING, TensorShape({})),
        Tensor(DT_STRING, TensorShape({})),
        Tensor(DT_STRING, TensorShape({})),
        Tensor(DT_STRING, TensorShape({columns_.size()}))};
    data->tensors_[0].scalar<string>()() = filename_;
    data->tensors_[1].scalar<string>()() = entryname_;
    data->tensors_[2].scalar<string>()() = filtername_;
    data->tensors_[3].scalar<string>()() = schema_;
    for (size_t i = 0; i < columns_.size(); i++) {
      data->tensors_[4].flat<string>()(i) = columns_[i];
    }

    EncodeAttributes(data);
  }
  bool Decode(const VariantTensorData& data) {
    filename_ = data.tensors(0).scalar<string>()();
    entryname_ = data.tensors(1).scalar<string>()();
    filtername_ = data.tensors(2).scalar<string>()();
    schema_ = data.tensors(3).scalar<string>()();
    columns_.resize(data.tensors(4).NumElements());
    for (int64 i = 0; i < data.tensors(4).NumElements(); i++) {
      columns_[i] = data.tensors_[4].flat<string>()(i);
    }

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
  const string& schema() const {
    return schema_;
  }
  const std::vector<string>& columns() const {
    return columns_;
  }
 protected:
  virtual Status FromStream(io::InputStreamInterface* s) = 0;

  virtual Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx,
    std::unique_ptr<T>& state, int64 record_to_read, int64* record_read,
    std::vector<Tensor>* out_tensors) const = 0;

  virtual void EncodeAttributes(VariantTensorData* data) const = 0;

  virtual bool DecodeAttributes(const VariantTensorData& data) = 0;

  Status ReadInputStream(io::InputStreamInterface* s, int64 chunk, int64 count,
    string* buffer, int64* returned) const {

    int64 offset = s->Tell();
    int64 bytes_to_read = count * chunk;
    Status status = (buffer == nullptr) ? s->SkipNBytes(bytes_to_read) : s->ReadNBytes(bytes_to_read, buffer);
    if (!(status.ok() || status == errors::OutOfRange("EOF reached"))) {
      return status;
    }
    int64 bytes_read = s->Tell() - offset;
    if (bytes_read % chunk != 0) {
      return errors::DataLoss("corrupted data, expected multiple of ", chunk, ", received ", bytes_read);
    }
    *returned = bytes_read / chunk;
    return Status::OK();
  }
  Status ReadReferenceRecord(void* s, IteratorContext* ctx, std::unique_ptr<T>& state,
    int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {

    return ReadRecord(static_cast<io::InputStreamInterface*>(s), ctx, state, record_to_read,
      record_read, out_tensors);
  }
  string filename_;
  string entryname_;
  string filtername_;
  string schema_;
  std::vector<string> columns_;
};

template<typename T>
class FileInputOp: public OpKernel {
 public:
  explicit FileInputOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("filters", &filters_));
    OP_REQUIRES_OK(context, context->GetAttr("columns", &columns_));
    OP_REQUIRES_OK(context, context->GetAttr("schema", &schema_));
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
        uint64 size = 0;
        OP_REQUIRES_OK(ctx, env_->GetFileSize(filename, &size));
        SizedRandomAccessFileStream file_stream(file.get(), size);
        T entry;
        OP_REQUIRES_OK(ctx, entry.FromInputStream(&file_stream, filename, string(""), string(""), schema_, columns_));
        output.emplace_back(std::move(entry));
        continue;
      }

      std::unique_ptr<struct archive, void(*)(struct archive *)> archive(archive_read_new(), [](struct archive *a){ archive_read_free(a);});
      OP_REQUIRES_OK(ctx, ArchiveInputStream::SetupFilters(archive.get(), filters_));

      ArchiveInputStream archive_stream(file.get(), archive.get());

      OP_REQUIRES(
          ctx, (archive_read_open(archive.get(), &archive_stream, NULL, ArchiveInputStream::CallbackRead, NULL) == ARCHIVE_OK),
          errors::InvalidArgument("unable to open datainput for ", filename, ": ", archive_error_string(archive.get())));

      size_t index = output.size();

      struct archive_entry *entry;
      while (archive_read_next_header(archive.get(), &entry) == ARCHIVE_OK) {
        string entryname = archive_entry_pathname(entry);
	string filtername;
        if (ArchiveInputStream::MatchFilters(archive.get(), entryname, filters_, &filtername)) {
	  T entry;
	  if (filtername == "none") {
            // If filter is none, then just use the initial stream.
	    // NOTE: Looks like libarchive may not be able to handle
	    // none with text type correctly (not reading data in none archive)
	    // So use the shortcut here.
            uint64 size = 0;
            OP_REQUIRES_OK(ctx, env_->GetFileSize(filename, &size));
            SizedRandomAccessFileStream file_stream(file.get(), size);
            OP_REQUIRES_OK(ctx, entry.FromInputStream(&file_stream, filename, entryname, filtername, schema_, columns_));
	  } else if (filtername == "gz") {
            // Treat gz file specially. Looks like libarchive always have issue
            // with text file so use ZlibInputStream. Now libarchive
            // is mostly used for archive (not compressio).
            io::RandomAccessInputStream file_stream(file.get());
            io::ZlibCompressionOptions zlib_compression_options = zlib_compression_options = io::ZlibCompressionOptions::GZIP();
            io::ZlibInputStream compression_stream(&file_stream, 65536, 65536,  zlib_compression_options);
            OP_REQUIRES_OK(ctx, entry.FromInputStream(&compression_stream, filename, entryname, filtername, schema_, columns_));
          } else {
            archive_stream.ResetEntryOffset();
            OP_REQUIRES_OK(ctx, entry.FromInputStream(&archive_stream, filename, entryname, filtername, schema_, columns_));
          }
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
  std::vector<string> columns_ GUARDED_BY(mu_);
  string schema_ GUARDED_BY(mu_);
};

template<typename InputType, typename StateType>
class FileInputDatasetBase : public DatasetBase {
 public:
  FileInputDatasetBase(OpKernelContext* ctx, const std::vector<InputType>& input,
    const int64 batch, const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        ctx_(ctx),
        input_(input),
        batch_(batch),
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
    Node* batch_node;
    Tensor batch_tensor(DT_INT64, TensorShape({}));
    batch_tensor.scalar<int64>()() = batch_;
    TF_RETURN_IF_ERROR(b->AddTensor(batch_tensor, &batch_node));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_node, batch_node}, node));
    return Status::OK();
  }
 private:
  class Iterator : public DatasetIterator<FileInputDatasetBase<InputType, StateType>> {
   public:
    using tensorflow::data::DatasetIterator<FileInputDatasetBase<InputType, StateType>>::dataset;
    explicit Iterator(const typename tensorflow::data::DatasetIterator<FileInputDatasetBase<InputType, StateType>>::Params& params)
        : DatasetIterator<FileInputDatasetBase<InputType, StateType>>(params), stream_(nullptr), archive_(nullptr, [](struct archive *a){ archive_read_free(a);}), file_(nullptr){}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      int64 returned = 0;
      int64 count = dataset()->batch_ == 0 ? 1 : dataset()->batch_;
      while (returned < count) {
        if (stream_) {
          TF_RETURN_IF_ERROR(dataset()->input_[current_input_index_].ReadBatchRecord(stream_.get(), ctx, current_input_state_, dataset()->batch_, count, &returned, out_tensors));
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
          if (out_tensors->size() != 0) {
            *end_of_sequence = false;
            return Status::OK();
          }
          *end_of_sequence = true;
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      };
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
      if (filtername.size() == 0 || filtername == "none") {
	// If filter is none, then just use the initial stream.
	// NOTE: Looks like libarchive may not be able to handle
	// none with text type correctly (not reading data in none archive)
	// So use the shortcut here.
        uint64 size = 0;
        TF_RETURN_IF_ERROR(env->GetFileSize(filename, &size));
        stream_.reset(new SizedRandomAccessFileStream(file_.get(), size));
        return Status::OK();
      } else if (filtername == "gz") {
        // Treat gz file specially. Looks like libarchive always have issue
	// with text file so use ZlibInputStream. Now libarchive
	// is mostly used for archive (not compressio).
	io::ZlibCompressionOptions zlib_compression_options = zlib_compression_options = io::ZlibCompressionOptions::GZIP();
        file_stream_.reset(new io::RandomAccessInputStream(file_.get()));
	stream_.reset(new io::ZlibInputStream(file_stream_.get(), 65536, 65536,  zlib_compression_options));
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
      file_stream_.reset(nullptr);
      file_.reset(nullptr);
    }

    mutex mu_;
    size_t current_input_index_ GUARDED_BY(mu_) = 0;
    std::unique_ptr<StateType> current_input_state_ GUARDED_BY(mu_);
    std::unique_ptr<io::InputStreamInterface> stream_ GUARDED_BY(mu_);
    std::unique_ptr<struct archive, void(*)(struct archive *)> archive_ GUARDED_BY(mu_);
    std::unique_ptr<io::InputStreamInterface> file_stream_ GUARDED_BY(mu_);
    std::unique_ptr<tensorflow::RandomAccessFile> file_ GUARDED_BY(mu_);
  };
  OpKernelContext* ctx_;
 protected:
  std::vector<InputType> input_;
  int64 batch_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

template<typename InputType, typename StateType>
class FileInputDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit FileInputDatasetOp(OpKernelConstruction* ctx)
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
    const Tensor* batch_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("batch", &batch_tensor));
    int64 batch = batch_tensor->scalar<int64>()();
    *output = new FileInputDatasetBase<InputType, StateType>(ctx, input, batch, output_types_, output_shapes_);
  }
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

template<typename T>
class StreamInput : public DataInput<T> {
 public:
  Status FromInputEndpoint(const string& endpoint, const string& schema,
    const std::vector<string>& columns) {

    endpoint_ = endpoint;
    schema_ = schema;
    columns_ = columns;
    return FromEndpoint(endpoint);
  }
  void Encode(VariantTensorData* data) const {
    data->tensors_ = {
        Tensor(DT_STRING, TensorShape({})),
        Tensor(DT_STRING, TensorShape({})),
        Tensor(DT_STRING, TensorShape({columns_.size()}))};
    data->tensors_[0].scalar<string>()() = endpoint_;
    data->tensors_[1].scalar<string>()() = schema_;
    for (size_t i = 0; i < columns_.size(); i++) {
      data->tensors_[2].flat<string>()(i) = columns_[i];
    }
    EncodeAttributes(data);
  }
  bool Decode(const VariantTensorData& data) {
    endpoint_ = data.tensors(0).scalar<string>()();
    schema_ = data.tensors(1).scalar<string>()();
    columns_.resize(data.tensors(2).NumElements());
    for (int64 i = 0; i < data.tensors(2).NumElements(); i++) {
      columns_[i] = data.tensors_[2].flat<string>()(i);
    }
    return DecodeAttributes(data);
  }
  const string& endpoint() const {
    return endpoint_;
  }
  const string& schema() const {
    return schema_;
  }
  const std::vector<string>& columns() const {
    return columns_;
  }
  Status ReadBatchRecord(IteratorContext* ctx, std::unique_ptr<T>& state,
    int64 batch, int64 count, int64* returned, std::vector<Tensor>* out_tensors) const {

    return (static_cast<const DataInput<T> *>(this))->ReadReferenceBatchRecord(
      nullptr, ctx, state, batch, count, returned, out_tensors);
  }
 protected:
  virtual Status FromEndpoint(const string& endpoint) = 0;
  virtual Status ReadRecord(IteratorContext* ctx, std::unique_ptr<T>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const = 0;
  virtual void EncodeAttributes(VariantTensorData* data) const = 0;
  virtual bool DecodeAttributes(const VariantTensorData& data) = 0;
  Status ReadReferenceRecord(void* s, IteratorContext* ctx, std::unique_ptr<T>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    return ReadRecord(ctx, state, record_to_read, record_read, out_tensors);
  }
  string endpoint_;
  string schema_;
  std::vector<string> columns_;
};

template<typename T>
class StreamInputOp: public OpKernel {
 public:
  explicit StreamInputOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("columns", &columns_));
    OP_REQUIRES_OK(context, context->GetAttr("schema", &schema_));
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

    for (const auto& endpoint: source) {
      T entry;
      OP_REQUIRES_OK(ctx, entry.FromInputEndpoint(endpoint, schema_, columns_));
      output.emplace_back(std::move(entry));
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
  string schema_ GUARDED_BY(mu_);
  std::vector<string> columns_ GUARDED_BY(mu_);
};
template<typename InputType, typename StateType>
class StreamInputDatasetBase : public DatasetBase {
 public:
  StreamInputDatasetBase(OpKernelContext* ctx, const std::vector<InputType>& input,
    const int64 batch, const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        ctx_(ctx),
        input_(input),
        batch_(batch),
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
    Node* batch_node;
    Tensor batch_tensor(DT_INT64, TensorShape({}));
    batch_tensor.scalar<int64>()() = batch_;
    TF_RETURN_IF_ERROR(b->AddTensor(batch_tensor, &batch_node));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_node, batch_node}, node));
    return Status::OK();
  }
 private:
  class Iterator : public DatasetIterator<StreamInputDatasetBase<InputType, StateType>> {
   public:
    using tensorflow::data::DatasetIterator<StreamInputDatasetBase<InputType, StateType>>::dataset;
    explicit Iterator(const typename tensorflow::data::DatasetIterator<StreamInputDatasetBase<InputType, StateType>>::Params& params)
        : DatasetIterator<StreamInputDatasetBase<InputType, StateType>>(params) {}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      int64 returned = 0;
      int64 count = dataset()->batch_ == 0 ? 1 : dataset()->batch_;
      while (returned < count) {
        if (current_input_index_ < dataset()->input_.size()) {
          TF_RETURN_IF_ERROR(dataset()->input_[current_input_index_].ReadBatchRecord(
            ctx, current_input_state_, dataset()->batch_, count, &returned, out_tensors));

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
          if (out_tensors->size() != 0) {
            *end_of_sequence = false;
            return Status::OK();
          }
          *end_of_sequence = true;
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      };
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
      current_input_state_.reset(nullptr);

      return Status::OK();
    }

    // Resets all streams.
    void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      current_input_state_.reset(nullptr);
    }

    mutex mu_;
    size_t current_input_index_ GUARDED_BY(mu_) = 0;
    std::unique_ptr<StateType> current_input_state_ GUARDED_BY(mu_);
  };
  OpKernelContext* ctx_;
 protected:
  std::vector<InputType> input_;
  int64 batch_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

template<typename InputType, typename StateType>
class StreamInputDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit StreamInputDatasetOp(OpKernelConstruction* ctx)
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
    const Tensor* batch_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("batch", &batch_tensor));
    int64 batch = batch_tensor->scalar<int64>()();
    *output = new StreamInputDatasetBase<InputType, StateType>(ctx, input, batch, output_types_, output_shapes_);
  }
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};
}  // namespace data
}  // namespace tensorflow
