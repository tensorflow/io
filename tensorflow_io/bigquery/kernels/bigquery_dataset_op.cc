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

#include "tensorflow_io/bigquery/kernels/bigquery_lib.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace data {
namespace {

class BigQueryDatasetOp : public DatasetOpKernel {
 public:
  explicit BigQueryDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("selected_fields", &selected_fields_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
  }
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    string stream;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "stream", &stream));
    string avro_schema;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<string>(ctx, "avro_schema", &avro_schema));

    BigQueryClientResource* client_resource;
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

    *output =
        new Dataset(ctx, client_resource, output_types_vector,
                    std::move(output_shapes), std::move(stream),
                    std::move(avro_schema), selected_fields_, output_types_);
  }

 private:
  std::vector<string> selected_fields_;
  std::vector<DataType> output_types_;

  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx,
                     tensorflow::BigQueryClientResource* client_resource,
                     const DataTypeVector& output_types_vector,
                     std::vector<PartialTensorShape> output_shapes,
                     string stream,
                     string avro_schema,
                     std::vector<string> selected_fields,
                     std::vector<DataType> output_types)
        : DatasetBase(DatasetContext(ctx)),
          client_resource_(client_resource),
          output_types_vector_(output_types_vector),
          output_shapes_(std::move(output_shapes)),
          stream_(stream),
          selected_fields_(selected_fields),
          output_types_(output_types),
          schema_(absl::make_unique<avro::ValidSchema>())  {
      client_resource_->Ref();
      std::istringstream istream(avro_schema);
      avro::compileJsonSchema(istream, *schema_);
    }

    ~Dataset() override { client_resource_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::BigQueryScan")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_vector_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    const string& stream() const { return stream_; }

    const std::vector<string>& selected_fields() const {
      return selected_fields_;
    }

    const std::vector<DataType>& output_types() const { return output_types_; }

    const std::unique_ptr<avro::ValidSchema>& schema() const { return schema_; }

    string DebugString() const override {
      return "BigQueryScanDatasetOp::Dataset";
    }

    tensorflow::BigQueryClientResource* client_resource() const {
      return client_resource_;
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      return errors::Unimplemented("%s does not support serialization",
                                   DebugString());
    }

   private:
    class Iterator : public BigQueryReaderDatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : BigQueryReaderDatasetIterator<Dataset>(params) {
      }
    };

    tensorflow::BigQueryClientResource* client_resource_;
    const DataTypeVector output_types_vector_;
    const std::vector<PartialTensorShape> output_shapes_;
    const string stream_;
    const std::vector<string> selected_fields_;
    const std::vector<DataType> output_types_;
    const std::unique_ptr<avro::ValidSchema> schema_;
  };
};

REGISTER_KERNEL_BUILDER(Name("BigQueryDataset").Device(DEVICE_CPU),
                        BigQueryDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
