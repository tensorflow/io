#include <ctime>
#include <iostream>
#include <orc/Exceptions.hh>
#include <orc/OrcFile.hh>
#include <orc/Reader.hh>
#include <orc/Type.hh>

#include "orc/orc-config.hh"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {
namespace data {

class ORCReadable : public IOReadableInterface {
 public:
  ORCReadable(Env* env) : env_(env) {}
  ~ORCReadable() {}
  Status Init(const std::vector<string>& input,
              const std::vector<string>& metadata, const void* memory_data,
              const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 filename is not supported");
    }
    const string& filename = input[0];
    // read packet data
    orc::RowReaderOptions row_reader_opts;
    orc::ReaderOptions reader_opts;
    std::unique_ptr<orc::Reader> reader =
        orc::createReader(orc::readFile(filename), reader_opts);

    row_reader_ = reader->createRowReader(row_reader_opts);
    LOG(INFO) << "ORC file schema:" << reader->getType().toString();

    // Parse columns. We assume the orc record file is a flat array
    auto row_count = reader->getNumberOfRows();
    for (int i = 0; i < reader->getType().getSubtypeCount(); ++i) {
      auto field_name = reader->getType().getFieldName(i);
      auto subtype = reader->getType().getSubtype(i);
      DataType dtype;
      switch (static_cast<int64_t>(subtype->getKind())) {
        case orc::STRING:
          dtype = DT_STRING;
          break;
        case orc::DOUBLE:
          dtype = DT_DOUBLE;
          break;
        case orc::FLOAT:
          dtype = DT_DOUBLE;
          break;
        default:
          return errors::InvalidArgument("data type is not supported: ",
                                         subtype->toString());
      }
      columns_.push_back(field_name);
      shapes_.push_back(TensorShape({static_cast<int64>(row_count)}));
      dtypes_.push_back(dtype);
      columns_index_[field_name] = i;
      tensors_.emplace_back(
          Tensor(dtype, TensorShape({static_cast<int64>(row_count)})));
    }
    // Fill in the values
    std::unique_ptr<orc::ColumnVectorBatch> batch =
        row_reader_->createRowBatch(10);
    auto* fields = dynamic_cast<orc::StructVectorBatch*>(batch.get());
    int64_t record_index = 0;
    while (row_reader_->next(*batch)) {
      for (uint32_t r = 0; r < batch->numElements; ++r) {
        for (size_t column_index = 0; column_index < columns_.size();
             column_index++) {
          if (dtypes_[column_index] == DT_DOUBLE) {
            auto* float_col = dynamic_cast<orc::DoubleVectorBatch*>(
                fields->fields[column_index]);
            double* buffer1 = float_col->data.data();
            tensors_[column_index].flat<double>()(record_index) = buffer1[r];
          } else if (dtypes_[column_index] == DT_STRING) {
            auto* string_col = dynamic_cast<orc::StringVectorBatch*>(
                fields->fields[column_index]);
            char** buffer = string_col->data.data();
            int64_t* lengths = string_col->length.data();
            tensors_[column_index].flat<tstring>()(record_index) =
                std::string(buffer[r], lengths[r]);
          }
        }
        record_index++;
      }
    }

    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop, const string& component,
              int64* record_read, Tensor* value, Tensor* label) override {
    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];

    (*record_read) = 0;
    if (start >= shapes_[column_index].dim_size(0)) {
      return Status::OK();
    }
    const string& column = component;
    int64 element_start = start < shapes_[column_index].dim_size(0)
                              ? start
                              : shapes_[column_index].dim_size(0);
    int64 element_stop = stop < shapes_[column_index].dim_size(0)
                             ? stop
                             : shapes_[column_index].dim_size(0);
    if (element_start > element_stop) {
      return errors::InvalidArgument("dataset ", column,
                                     " selection is out of boundary");
    }
    if (element_start == element_stop) {
      return Status::OK();
    }
    for (int i = element_start; i < element_stop; i++) {
      if (dtypes_[column_index] == DT_STRING) {
        value->flat<tstring>().data()[i] =
            tensors_[column_index].flat<tstring>().data()[i];
        // LOG(INFO) << " column_index: " << column_index << " index: " << i <<
        // " data: " << value->flat<tstring>().data()[i];
      } else if (dtypes_[column_index] == DT_DOUBLE) {
        value->flat<double>().data()[i] =
            tensors_[column_index].flat<double>().data()[i];
        // LOG(INFO) << " column_index: " << column_index << " index: " << i <<
        // " data: " << value->flat<double>().data()[i];
      } else {
        return errors::InvalidArgument("invalid data type: ",
                                       dtypes_[column_index]);
      }
    }
    (*record_read) = element_stop - element_start;

    return Status::OK();
  }

  Status Components(std::vector<string>* components) override {
    components->clear();
    for (size_t i = 0; i < columns_.size(); i++) {
      components->push_back(columns_[i]);
    }
    return Status::OK();
  }

  Status Spec(const string& component, PartialTensorShape* shape,
              DataType* dtype, bool label) override {
    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];
    *shape = shapes_[column_index];
    *dtype = dtypes_[column_index];
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("ORCReadable");
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  std::unique_ptr<orc::RowReader> row_reader_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
  std::vector<Tensor> tensors_;

  std::vector<DataType> dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
};
REGISTER_KERNEL_BUILDER(Name("IO>ORCReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<ORCReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>ORCReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<ORCReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>ORCReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<ORCReadable>);
}  // namespace data
}  // namespace tensorflow