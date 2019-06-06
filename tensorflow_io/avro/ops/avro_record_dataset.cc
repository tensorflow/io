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

#include <avro.h>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"

// As boiler plate I used
// https://github.com/tensorflow/tensorflow/core/kernels/reader_dataset_ops.cc
// https://github.com/tensorflow/tensorflow/blob/v1.4.1/tensorflow/core/ops/dataset_ops.cc
// (register op)

namespace tensorflow {

// Register the avro record dataset operator
REGISTER_OP("AvroRecordDataset")
    .Input("filenames: string")
    .Input("schema: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that emits the avro records from one or more files.
filenames: A scalar or vector containing the name(s) of the file(s) to be
  read.
schema: A string used that is used for schema resolution.
)doc");

// This class represents the avro reader options
class AvroReaderOptions {
 public:
  // Creates avro reader options with the given schema and buffer size.
  //
  static AvroReaderOptions CreateAvroReaderOptions(const string& schema,
                                                   int64 buffer_size) {
    AvroReaderOptions options;
    options.schema = schema;
    options.buffer_size = buffer_size;
    return options;
  }
  string schema;
  int64 buffer_size =
      256 * 1024;  // 256 kB as default but this can be overwritten by the user
};

void AvroFileReaderDestructor(avro_file_reader_t reader) {
  // I don't think we need the CHECK_NOTNULL
  CHECK_GE(avro_file_reader_close(reader), 0);
}

void AvroSchemaDestructor(avro_schema_t schema) {
  // Confusingly, it appears that the avro_file_reader_t creates its
  // own reference to this schema, so the schema is not really
  // "uniquely" owned...
  CHECK_GE(avro_schema_decref(schema), 0);
};

void AvroValueInterfaceDestructor(avro_value_iface_t * iface)  {
  avro_value_iface_decref(iface);
}


// This reader is not thread safe
class SequentialAvroRecordReader {
 public:
  // Construct a sequential avro record reader
  //
  // 'file' is the random access file
  //
  // 'file_size' is the size of the file
  //
  // 'filename' is the name of the file
  //
  // 'options' are avro reader options
  //
  SequentialAvroRecordReader(RandomAccessFile* file, const uint64 file_size,
                             const string& filename,
                             const AvroReaderOptions& options =
                                 AvroReaderOptions())
    : initialized_(false),
      filename_(filename),
      file_buffer_(file_size, '\0'),
      input_buffer_size_(options.buffer_size),
      input_buffer_(new io::InputBuffer(file, options.buffer_size)),
      reader_schema_str_(options.schema),
      file_reader_(nullptr, AvroFileReaderDestructor),
      reader_schema_(nullptr, AvroSchemaDestructor),
      writer_schema_(nullptr, AvroSchemaDestructor),
      p_reader_iface_(nullptr, AvroValueInterfaceDestructor),
      p_writer_iface_(nullptr, AvroValueInterfaceDestructor) { }
  virtual ~SequentialAvroRecordReader() {
    // Guard against clean-up of non-initialized instances
    if (initialized_) {
      avro_value_decref(&reader_value_);
      avro_value_decref(&writer_value_);
    }
  }
  // Reads the next record into the string record
  //
  // 'record' pointer to the string where to load the record in
  //
  // returns Status about this operation
  //
  Status ReadRecord(string* record) {
    bool at_end =
      avro_file_reader_read_value(file_reader_.get(), &writer_value_) != 0;
    // Are writer_value_ and reader_value_ aliases to the same thing???
    size_t len;
    if (avro_value_sizeof(&reader_value_, &len)) {
      return Status(errors::InvalidArgument("Could not find size of value, ",
                                            avro_strerror()));
    }
    record->resize(len);
    avro_writer_t mem_writer = avro_writer_memory(record->data(), len);
    if (avro_value_write(mem_writer, &reader_value_)) {
      avro_writer_free(mem_writer);
      return Status(errors::InvalidArgument("Unable to write value to memory."));
    }
    avro_writer_free(mem_writer);
    return at_end ? errors::OutOfRange("eof") : Status::OK();
  }
  // Call for startup of work after construction. Loads data into memory and
  // sets up the avro file reader
  //
  // returns Status about this operation
  //
  Status OnWorkStartup() {
    // Clear the error message, so we won't get a wrong message
    avro_set_error("");
    Status status;

    // Read the file into memory via the gfile API so we can accept
    // files on S3, HDFS, etc.
    TF_RETURN_IF_ERROR(CreateAndLoadFileIntoBuffer(input_buffer_size_));
    FILE* fp = fmemopen(static_cast<void*>(const_cast<char*>(file_buffer_.data())),
                        file_buffer_.size(), "r");
    if (fp == nullptr) {
      return Status(errors::InvalidArgument("Unable to open file ", filename_,
                                            " on memory in avro reader."));
    }

    // Get an avro file reader for that file handle, the 1 indicates to close
    // the file handle when done
    avro_file_reader_t file_reader_tmp;
    if (avro_file_reader_fp(fp, filename_.c_str(), 1, &file_reader_tmp) != 0) {
      return Status(errors::InvalidArgument("Unable to open file ", filename_,
                                            " in avro reader. ", avro_strerror()));
    }
    file_reader_.reset(file_reader_tmp);

    writer_schema_.reset(avro_file_reader_get_writer_schema(file_reader_.get()));

    // The user provided a schema for the reader, check if we need to do schema
    // resolution
    bool do_resolution = false;
    if (reader_schema_str_.length() > 0) {

      avro_schema_t reader_schema_tmp;
      // Create value to read into using the provided schema
      if (avro_schema_from_json_length(reader_schema_str_.data(),
                                       reader_schema_str_.length(),
                                       &reader_schema_tmp) != 0) {
        return Status(errors::InvalidArgument(
            "The provided json schema is invalid. ", avro_strerror()));
      }
      reader_schema_.reset(reader_schema_tmp);
      do_resolution = !avro_schema_equal(writer_schema_.get(), reader_schema_.get());
      // We need to do a schema resolution, if the schemas are not the same
    }

    if (do_resolution) {
      // Create reader class
      p_reader_iface_.reset(avro_generic_class_from_schema(reader_schema_.get()));
      // Create instance for reader class
      if (avro_generic_value_new(p_reader_iface_.get(), &reader_value_) != 0) {
        return Status(errors::InvalidArgument(
            "Unable to value for user-supplied schema. ", avro_strerror()));
      }
      // Create resolved writer class
      p_writer_iface_.reset(avro_resolved_writer_new(writer_schema_.get(), reader_schema_.get()));
      if (p_writer_iface_.get() == nullptr) {
        // Cleanup
        avro_value_decref(&reader_value_);
        return Status(errors::InvalidArgument("Schemas are incompatible. ",
                                              avro_strerror()));
      }
      // Create instance for resolved writer class
      if (avro_resolved_writer_new_value(p_writer_iface_.get(), &writer_value_) !=
          0) {
        // Cleanup
        avro_value_decref(&reader_value_);
        return Status(
            errors::InvalidArgument("Unable to create resolved writer."));
      }
      avro_resolved_writer_set_dest(&writer_value_, &reader_value_);
    } else {
      p_writer_iface_.reset(avro_generic_class_from_schema(writer_schema_.get()));
      if (avro_generic_value_new(p_writer_iface_.get(), &writer_value_) != 0) {
        return Status(errors::InvalidArgument(
            "Unable to create instance for generic class."));
      }
      // The reader_value_ is the same as the writer_value_ in the case we do
      // not need to resolve the schema
      avro_value_copy_ref(&reader_value_, &writer_value_);
    }

    // We initialized this avro record reader
    initialized_ = true;

    return Status::OK();
  }

 private:
  // Loads file contents into file_buffer_
  //
  // 'read_buffer_size' buffer size when reading file contents
  //
  Status CreateAndLoadFileIntoBuffer(int64 read_buffer_size) {
    int64 total_bytes_read = 0;
    Status status;

    // While we still need to read data
    char* buffer = const_cast<char*>(file_buffer_.data());
    while (total_bytes_read < file_buffer_.size()) {
      size_t bytes_read;
      status = input_buffer_->ReadNBytes(read_buffer_size, buffer,
                                         &bytes_read);
      total_bytes_read += bytes_read;
      buffer += bytes_read;
      // If we are at the end of the file
      if (errors::IsOutOfRange(status)) {
        break;
      } else if (!status.ok()) {
        return status;
      }
    }

    CHECK_EQ(total_bytes_read, file_buffer_.size());
    return Status::OK();
  }

  bool initialized_;                               // Has been initialized
  const string filename_;                                // Name of the file
  std::string file_buffer_;                        // The data buffer
  const size_t input_buffer_size_;
  std::unique_ptr<io::InputBuffer> input_buffer_;  // input buffer used to load
                                                   // from random access file
  const string reader_schema_str_;  // User supplied string to read this avro
                                    // file

  using AvroFileReaderUPtr = std::unique_ptr<struct avro_file_reader_t_,
                                             void(*)(avro_file_reader_t)>;
  AvroFileReaderUPtr file_reader_;  // Avro file reader
  // TODO: Use std::remove_pointer
  using AvroSchemaUPtr = std::unique_ptr<struct avro_obj_t,
                                         void(*)(avro_schema_t)>;
  AvroSchemaUPtr reader_schema_;  // Schema to read, set only when doing schema
                                  // resolution
  AvroSchemaUPtr writer_schema_; // Schema that the file was written with
  using AvroValueInterfacePtr = std::unique_ptr<avro_value_iface_t,
                                                void(*)(avro_value_iface_t*)>;
  AvroValueInterfacePtr p_reader_iface_;  // Reader class info to create instances
  AvroValueInterfacePtr p_writer_iface_;  // Writer class info to create instances
  avro_value_t reader_value_;  // Reader value, unequal from writer value when
                               // doing schema resolution
  avro_value_t writer_value_;  // Writer value
};

class AvroRecordDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

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

    string schema;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "schema", &schema));

    int64 buffer_size = -1;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(ctx, buffer_size >= 256,
                errors::InvalidArgument("`buffer_size` must be >= 256 B"));

    *output = new Dataset(ctx, std::move(filenames), schema, buffer_size);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, std::vector<string> filenames,
                     const string& schema, int64 buffer_size)
        : DatasetBase(DatasetContext(ctx)), filenames_(std::move(filenames)),
          options_(AvroReaderOptions::CreateAvroReaderOptions(schema,
                                                              buffer_size)) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const
        override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::AvroRecord")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override { return "AvroRecordDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** node) const override {
      return errors::Unimplemented("%s does not support serialization",
                                   DebugString());
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
        do { // What is the point of this loop???
          // We are currently processing a file, so try to read the next record.
          if (reader_) {
            Tensor result_tensor(cpu_allocator(), DT_STRING, {});
            Status s = reader_->ReadRecord(&result_tensor.scalar<string>()());
            if (s.ok()) {
              out_tensors->emplace_back(std::move(result_tensor));
              *end_of_sequence = false;
              return Status::OK();
            } else if (!errors::IsOutOfRange(s)) {
              return s;
            } else {
              CHECK(errors::IsOutOfRange(s));
              // We have reached the end of the current file, so maybe
              // move on to next file.
              reader_.reset();
              file_.reset();
              ++current_file_index_;
            }
          }

          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          // Actually move on to next file.
          // Looks like this cannot request multiple files in parallel. Hmm.
          const string& next_filename =
              dataset()->filenames_[current_file_index_];

          TF_RETURN_IF_ERROR(
            ctx->env()->NewRandomAccessFile(next_filename, &file_));

          uint64 file_size;
          TF_RETURN_IF_ERROR(
            ctx->env()->GetFileSize(next_filename, &file_size));

          reader_.reset(new SequentialAvroRecordReader(
              file_.get(), file_size, next_filename, dataset()->options_));
          TF_RETURN_IF_ERROR(reader_->OnWorkStartup());
        } while (true);
      }

     private:
      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;

      // `reader_` will borrow the object that `file_` points to, so
      // we must destroy `reader_` before `file_`.
      std::unique_ptr<RandomAccessFile> file_ GUARDED_BY(mu_);
      std::unique_ptr<SequentialAvroRecordReader> reader_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
    AvroReaderOptions options_;
  };
};

REGISTER_KERNEL_BUILDER(Name("AvroRecordDataset").Device(DEVICE_CPU),
                        AvroRecordDatasetOp);

}  // namespace tensorflow
