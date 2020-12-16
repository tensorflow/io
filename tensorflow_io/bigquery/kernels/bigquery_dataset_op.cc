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

#include <memory>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/ipc/api.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/bigquery/kernels/bigquery_lib.h"

namespace tensorflow {
namespace data {
namespace {

class BigQueryDatasetOp : public DatasetOpKernel {
 public:
  explicit BigQueryDatasetOp(OpKernelConstruction *ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("selected_fields", &selected_fields_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("offset", &offset_));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES_OK(ctx, GetDataFormat(data_format_str, &data_format_));
  }
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override {
    tstring stream;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "stream", &stream));
    tstring schema;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "schema", &schema));
    OP_REQUIRES(ctx, !schema.empty(),
                errors::InvalidArgument("schema must be non-empty"));

    BigQueryClientResource *client_resource;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &client_resource));
    core::ScopedUnref scoped_unref(client_resource);
    const uint64 num_outputs = selected_fields_.size();
    std::vector<PartialTensorShape> output_shapes;
    output_shapes.reserve(num_outputs);
    DataTypeVector output_types_vector;
    output_types_vector.reserve(num_outputs);
    for (uint64 i = 0; i < num_outputs; ++i) {
      output_shapes.push_back({});
      output_types_vector.push_back(output_types_[i]);
    }

    *output = new Dataset(ctx, client_resource, output_types_vector,
                          std::move(output_shapes), std::move(stream),
                          std::move(schema), selected_fields_, output_types_,
                          offset_, data_format_);
  }

 private:
  std::vector<string> selected_fields_;
  std::vector<DataType> output_types_;
  int64 offset_;
  apiv1beta1::DataFormat data_format_;

  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext *ctx,
                     tensorflow::BigQueryClientResource *client_resource,
                     const DataTypeVector &output_types_vector,
                     std::vector<PartialTensorShape> output_shapes,
                     string stream, string schema,
                     std::vector<string> selected_fields,
                     std::vector<DataType> output_types, int64 offset_,
                     apiv1beta1::DataFormat data_format)
        : DatasetBase(DatasetContext(ctx)),
          client_resource_(client_resource),
          output_types_vector_(output_types_vector),
          output_shapes_(std::move(output_shapes)),
          stream_(stream),
          selected_fields_(selected_fields),
          output_types_(output_types),
          offset_(offset_),
          avro_schema_(absl::make_unique<avro::ValidSchema>()),
          data_format_(data_format) {
      client_resource_->Ref();

      if (data_format == apiv1beta1::DataFormat::AVRO) {
        std::istringstream istream(schema);
        avro::compileJsonSchema(istream, *avro_schema_);
      } else if (data_format == apiv1beta1::DataFormat::ARROW) {
        auto buffer_ = std::make_shared<arrow::Buffer>(
            reinterpret_cast<const uint8_t *>(&schema[0]), schema.length());

        arrow::ipc::DictionaryMemo dict_memo;
        arrow::io::BufferReader input(buffer_);
        arrow::Result<std::shared_ptr<arrow::Schema>> result =
            arrow::ipc::ReadSchema(&input, &dict_memo);
        OP_REQUIRES(ctx, result.ok(),
                    errors::Internal("Error reading Arrow Schema",
                                     result.status().message()));
        arrow_schema_ = std::move(result).ValueUnsafe();
      } else {
        ctx->CtxFailure(errors::InvalidArgument("Invalid data_format"));
      }
    }

    ~Dataset() override { client_resource_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string &prefix) const override {
      if (data_format_ == apiv1beta1::DataFormat::AVRO) {
        return std::unique_ptr<IteratorBase>(
            new BigQueryReaderAvroDatasetIterator<Dataset>(
                {this, strings::StrCat(prefix, "::BigQueryAvroDataset")}));
      } else if (data_format_ == apiv1beta1::DataFormat::ARROW) {
        return std::unique_ptr<IteratorBase>(
            new BigQueryReaderArrowDatasetIterator<Dataset>(
                {this, strings::StrCat(prefix, "::BigQueryArrowDataset")}));
      }

      // Should never get there.
      throw std::exception();
    }

    const DataTypeVector &output_dtypes() const override {
      return output_types_vector_;
    }

    const std::vector<PartialTensorShape> &output_shapes() const override {
      return output_shapes_;
    }

    const string &stream() const { return stream_; }

    const std::vector<string> &selected_fields() const {
      return selected_fields_;
    }

    const std::vector<DataType> &output_types() const { return output_types_; }

    const std::unique_ptr<avro::ValidSchema> &avro_schema() const {
      return avro_schema_;
    }

    std::shared_ptr<::arrow::Schema> arrow_schema() const {
      return arrow_schema_;
    }

    const int64 offset() const { return offset_; }

    string DebugString() const override { return "BigQueryDatasetOp::Dataset"; }

    Status CheckExternalState() const override { return Status::OK(); }

    tensorflow::BigQueryClientResource *client_resource() const {
      return client_resource_;
    }

   protected:
    Status AsGraphDefInternal(SerializationContext *ctx,
                              DatasetGraphDefBuilder *b,
                              Node **output) const override {
      return errors::Unimplemented("%s does not support serialization",
                                   DebugString());
    }

   private:
    tensorflow::BigQueryClientResource *client_resource_;
    const DataTypeVector output_types_vector_;
    const std::vector<PartialTensorShape> output_shapes_;
    const string stream_;
    const std::vector<string> selected_fields_;
    const std::vector<DataType> output_types_;
    const std::unique_ptr<avro::ValidSchema> avro_schema_;
    const int64 offset_;
    std::shared_ptr<::arrow::Schema> arrow_schema_;
    const apiv1beta1::DataFormat data_format_;
  };
};

REGISTER_KERNEL_BUILDER(Name("IO>BigQueryDataset").Device(DEVICE_CPU),
                        BigQueryDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
