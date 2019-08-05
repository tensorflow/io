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
#include "tensorflow_io/arrow/kernels/arrow_kernels.h"
#include "arrow/io/api.h"
#include "arrow/ipc/feather.h"
#include "arrow/ipc/feather_generated.h"
#include "arrow/buffer.h"

namespace tensorflow {
namespace data {
namespace {

class ListFeatherColumnsOp : public OpKernel {
 public:
  explicit ListFeatherColumnsOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();
    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    // FEA1.....[metadata][uint32 metadata_length]FEA1
    static constexpr const char* kFeatherMagicBytes = "FEA1";

    size_t header_length = strlen(kFeatherMagicBytes);
    size_t footer_length = sizeof(uint32) + strlen(kFeatherMagicBytes);

    string buffer;
    buffer.resize(header_length > footer_length ? header_length : footer_length);

    StringPiece result;

    OP_REQUIRES_OK(context, file->Read(0, header_length, &result, &buffer[0]));
    OP_REQUIRES(context, !memcmp(buffer.data(), kFeatherMagicBytes, header_length), errors::InvalidArgument("not a feather file"));

    OP_REQUIRES_OK(context, file->Read(size - footer_length, footer_length, &result, &buffer[0]));
    OP_REQUIRES(context, !memcmp(buffer.data() + sizeof(uint32), kFeatherMagicBytes, footer_length - sizeof(uint32)), errors::InvalidArgument("incomplete feather file"));

    uint32 metadata_length = *reinterpret_cast<const uint32*>(buffer.data());

    buffer.resize(metadata_length);

    OP_REQUIRES_OK(context, file->Read(size - footer_length - metadata_length, metadata_length, &result, &buffer[0]));

    const ::arrow::ipc::feather::fbs::CTable* table = ::arrow::ipc::feather::fbs::GetCTable(buffer.data());

    OP_REQUIRES(context, (table->version() >= ::arrow::ipc::feather::kFeatherVersion), errors::InvalidArgument("feather file is old: ", table->version(), " vs. ", ::arrow::ipc::feather::kFeatherVersion));

    std::vector<string> columns;
    std::vector<string> dtypes;
    std::vector<int64> counts;
    columns.reserve(table->columns()->size());
    dtypes.reserve(table->columns()->size());
    counts.reserve(table->columns()->size());

    for (int64 i = 0; i < table->columns()->size(); i++) {
      DataType dtype = ::tensorflow::DataType::DT_INVALID;
      switch (table->columns()->Get(i)->values()->type()) {
      case ::arrow::ipc::feather::fbs::Type_BOOL:
        dtype = ::tensorflow::DataType::DT_BOOL;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT8:
        dtype = ::tensorflow::DataType::DT_INT8;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT16:
        dtype = ::tensorflow::DataType::DT_INT16;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT32:
        dtype = ::tensorflow::DataType::DT_INT32;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT64:
        dtype = ::tensorflow::DataType::DT_INT64;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT8:
        dtype = ::tensorflow::DataType::DT_UINT8;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT16:
        dtype = ::tensorflow::DataType::DT_UINT16;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT32:
        dtype = ::tensorflow::DataType::DT_UINT32;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT64:
        dtype = ::tensorflow::DataType::DT_UINT64;
        break;
      case ::arrow::ipc::feather::fbs::Type_FLOAT:
        dtype = ::tensorflow::DataType::DT_FLOAT;
        break;
      case ::arrow::ipc::feather::fbs::Type_DOUBLE:
        dtype = ::tensorflow::DataType::DT_DOUBLE;
        break;
      case ::arrow::ipc::feather::fbs::Type_UTF8:
      case ::arrow::ipc::feather::fbs::Type_BINARY:
      case ::arrow::ipc::feather::fbs::Type_CATEGORY:
      case ::arrow::ipc::feather::fbs::Type_TIMESTAMP:
      case ::arrow::ipc::feather::fbs::Type_DATE:
      case ::arrow::ipc::feather::fbs::Type_TIME:
      // case ::arrow::ipc::feather::fbs::Type_LARGE_UTF8:
      // case ::arrow::ipc::feather::fbs::Type_LARGE_BINARY:
      default:
        break;
      }
      columns.push_back(table->columns()->Get(i)->name()->str());
      dtypes.push_back(::tensorflow::DataTypeString(dtype));
      counts.push_back(table->num_rows());
    }

    TensorShape output_shape = filename_tensor.shape();
    output_shape.AddDim(columns.size());

    Tensor* columns_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &columns_tensor));
    Tensor* dtypes_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &dtypes_tensor));

    output_shape.AddDim(1);

    Tensor* shapes_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape, &shapes_tensor));

    for (size_t i = 0; i < columns.size(); i++) {
      columns_tensor->flat<string>()(i) = columns[i];
      dtypes_tensor->flat<string>()(i) = dtypes[i];
      shapes_tensor->flat<int64>()(i) = counts[i];
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("ListFeatherColumns").Device(DEVICE_CPU),
                        ListFeatherColumnsOp);


}  // namespace
}  // namespace data
}  // namespace tensorflow
