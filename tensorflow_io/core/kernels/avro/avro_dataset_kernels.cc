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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow_io/core/kernels/avro/utils/avro_file_stream_reader.h"

// As boiler plate I used
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/dataset.h
// DatasetBase
//
// Example build with headers
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/sql/BUILD
//
// dataset
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/dataset_ops.cc

// CSV parser
// Op definition:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/experimental_dataset_ops.cc
// Op implementation:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/experimental/csv_dataset_op.cc

// Example parser
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/parsing_ops.cc
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/experimental/parse_example_dataset_op.cc
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/example_proto_fast_parsing.cc
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/example_proto_helper.h

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/parsing_ops.py
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/data/ops/dataset_ops.py

// Attr vs Input
// https://www.tensorflow.org/guide/extend/op#attrs

// Batching
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/dataset_ops.cc
// BatchDataset
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/batch_dataset_op.cc
// https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/data/ops/dataset_ops.py
// BatchDataset

namespace tensorflow {
namespace data {
namespace {

class AvroDatasetOp : public DatasetOpKernel {
 public:
  explicit AvroDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reader_schema", &reader_schema_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_keys", &sparse_keys_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_keys", &dense_keys_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_types", &sparse_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_types", &dense_types_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_shapes", &dense_shapes_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    // Get filenames
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));
    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<tstring>()(i));
    }

    int64 batch_size;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("batch_size must be greater than zero."));

    bool drop_remainder;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "drop_remainder", &drop_remainder));

    // Get keys
    OpInputList sparse_keys;
    OpInputList dense_keys;

    // Get dense default tensors
    OpInputList dense_default_tensors;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("dense_defaults", &dense_default_tensors));

    OP_REQUIRES(ctx, dense_default_tensors.size() == dense_keys_.size(),
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_default_tensors.size(), " vs. ", dense_keys_.size()));

    std::vector<Tensor> dense_defaults(dense_default_tensors.begin(),
                                       dense_default_tensors.end());

    for (int d = 0; d < dense_keys_.size(); ++d) {
      const Tensor& def_value = dense_defaults[d];
      OP_REQUIRES(
          ctx, def_value.dtype() == dense_types_[d],
          errors::InvalidArgument(
              "For key '", dense_keys_[d], "' ", "dense_defaults[", d,
              "].dtype() == ", DataTypeString(def_value.dtype()),
              " != dense_types[", d, "] == ", DataTypeString(dense_types_[d])));
    }

    // first all dense and then all sparse tensors according to their input
    // order
    std::map<string, int> key_to_output_index;
    for (int d = 0; d < dense_keys_.size(); ++d) {
      auto result = key_to_output_index.insert({dense_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          dense_keys_[d]));
    }

    for (int d = 0; d < sparse_keys_.size(); ++d) {
      auto result = key_to_output_index.insert({sparse_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          sparse_keys_[d]));
    }
    int i = 0;
    for (auto it = key_to_output_index.begin(); it != key_to_output_index.end();
         it++) {
      it->second = i++;
    }

    // Get buffer sizes
    int64 input_stream_buffer_size;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "input_stream_buffer_size",
                                            &input_stream_buffer_size));
    OP_REQUIRES(ctx, input_stream_buffer_size > 0,
                errors::InvalidArgument(
                    "input_stream_buffer_size must be greater than zero."));

    int64 avro_data_buffer_size;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "avro_data_buffer_size",
                                            &avro_data_buffer_size));
    OP_REQUIRES(ctx, avro_data_buffer_size > 0,
                errors::InvalidArgument(
                    "avro_data_buffer_size must be greater than zero."));

    *output =
        new Dataset(ctx, std::move(filenames), batch_size, drop_remainder,
                    reader_schema_, input_stream_buffer_size,
                    avro_data_buffer_size, dense_defaults, sparse_keys_,
                    dense_keys_, std::move(key_to_output_index), sparse_types_,
                    dense_types_, dense_shapes_, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> filenames,
            int64 batch_size, bool drop_remainder, string reader_schema,
            int64 input_stream_buffer_size, int64 avro_data_buffer_size,
            std::vector<Tensor> dense_defaults, std::vector<string> sparse_keys,
            std::vector<string> dense_keys,
            std::map<string, int> key_to_output_index,
            const DataTypeVector& sparse_types,
            const DataTypeVector& dense_types,
            const std::vector<PartialTensorShape>& dense_shapes,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(std::move(filenames)),
          reader_schema_(std::move(reader_schema)),
          dense_defaults_(std::move(dense_defaults)),
          sparse_keys_(std::move(sparse_keys)),
          dense_keys_(std::move(dense_keys)),
          key_to_output_index_(std::move(key_to_output_index)),
          sparse_types_(sparse_types),
          dense_types_(dense_types),
          dense_shapes_(dense_shapes),
          output_types_(output_types),
          output_shapes_(output_shapes),
          config_(BuildConfig(batch_size, drop_remainder,
                              input_stream_buffer_size, avro_data_buffer_size,
                              dense_keys_, dense_types_, dense_shapes_,
                              dense_defaults_, sparse_keys_, sparse_types_)) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Avro")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "AvroDatasetOp::Dataset"; }

    Status CheckExternalState() const override { return Status::OK(); }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;

      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));

      Node* batch_size = nullptr;

      TF_RETURN_IF_ERROR(b->AddScalar(config_.batch_size, &batch_size));

      Node* drop_remainder = nullptr;

      TF_RETURN_IF_ERROR(b->AddScalar(config_.drop_remainder, &drop_remainder));

      std::vector<Node*> dense_defaults_nodes;
      dense_defaults_nodes.reserve(dense_defaults_.size());

      for (const Tensor& dense_default : dense_defaults_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(dense_default, &node));
        dense_defaults_nodes.emplace_back(node);
      }

      Node* input_stream_buffer_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(config_.input_stream_buffer_size,
                                      &input_stream_buffer_size));

      Node* avro_data_buffer_size = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(config_.avro_data_buffer_size, &avro_data_buffer_size));

      AttrValue reader_schema_attr;
      AttrValue sparse_keys_attr;
      AttrValue dense_keys_attr;
      AttrValue sparse_types_attr;
      AttrValue dense_attr;
      AttrValue dense_shapes_attr;

      b->BuildAttrValue(reader_schema_, &reader_schema_attr);
      b->BuildAttrValue(sparse_keys_, &sparse_keys_attr);
      b->BuildAttrValue(dense_keys_, &dense_keys_attr);
      b->BuildAttrValue(sparse_types_, &sparse_types_attr);
      b->BuildAttrValue(dense_types_, &dense_attr);
      b->BuildAttrValue(dense_shapes_, &dense_shapes_attr);

      TF_RETURN_IF_ERROR(
          b->AddDataset(this,
                        {{0, filenames},
                         {1, batch_size},
                         {2, drop_remainder},
                         {4, input_stream_buffer_size},
                         {5, avro_data_buffer_size}},  // single tensor inputs
                        {{3, dense_defaults_nodes}},   // list tensor inputs
                        {{"reader_schema", reader_schema_attr},
                         {"sparse_keys", sparse_keys_attr},
                         {"dense_keys", dense_keys_attr},
                         {"sparse_types", sparse_types_attr},
                         {"dense_types", dense_attr},
                         {"dense_shapes",
                          dense_shapes_attr}},  // non-tensor inputs, attributes
                        output));
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

        // Loops over all files
        do {
          // We are currently processing a file, so try to read the next record.
          if (reader_) {
            Status s = Read(ctx, out_tensors);
            if (s.ok()) {
              *end_of_sequence = false;
              return Status::OK();
            } else if (!errors::IsOutOfRange(s)) {
              return s;
            } else {
              CHECK(errors::IsOutOfRange(s));
              // We have reached the end of the current file, so maybe
              // move on to next file.
              reader_.reset(nullptr);
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

          reader_.reset(new AvroFileStreamReader(ctx->env(), next_filename,
                                                 dataset()->reader_schema_,
                                                 dataset()->config_));

          // Check status and set reader to nullptr to avoid read on broken
          // reader_
          Status s = reader_->OnWorkStartup();
          if (!s.ok()) {
            reader_.reset(nullptr);
            return s;
          }

        } while (true);
      }
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal");
      }
      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "Iterator does not support 'RestoreInternal')");
      }

     private:
      Status Read(IteratorContext* ctx, std::vector<Tensor>* out_tensors) {
        // TODO(fraudies): Perf; check if we can initialize and re-use the
        // result struct
        AvroResult avro_result;
        TF_RETURN_IF_ERROR((*reader_).Read(&avro_result));

        // Check validity of the result and assign it to the out tensors
        (*out_tensors).resize(dataset()->key_to_output_index_.size());
        for (int d = 0; d < dataset()->dense_keys_.size(); ++d) {
          int output_index =
              dataset()->key_to_output_index_.at(dataset()->dense_keys_[d]);
          CHECK(avro_result.dense_values[d].dtype() ==
                dataset()->output_dtypes()[output_index])
              << "Got wrong type for AvroDataset for key '"
              << dataset()->dense_keys_[d] << "'"
              << " (expected "
              << DataTypeString(dataset()->output_dtypes()[output_index])
              << ", got " << DataTypeString(avro_result.dense_values[d].dtype())
              << ").";

          const AvroParseConfig::Dense& dense = dataset()->config_.dense[d];
          if (dense.variable_length) {
            // Use dense_defaults_ here because it does not include the batch
            // replicate
            CHECK(dataset()->dense_defaults_[d].NumElements() == 1)
                << "For key '" << dataset()->dense_keys_[d] << "'"
                << "dense_shape[" << d
                << "] is a variable length shape: " << dense.shape.DebugString()
                << ", therefore "
                << "def_value[" << d << "] must contain a single element ("
                << "the padding element).  But its shape is: "
                << dense.default_value.shape().DebugString();

          } else {
            CHECK(dataset()->output_shapes_[output_index].IsCompatibleWith(
                avro_result.dense_values[d].shape()))
                << "Got wrong shape for key '" << dataset()->dense_keys_[d]
                << "' "
                << " (expected "
                << dataset()->output_shapes_[output_index].DebugString()
                << ", got " << avro_result.dense_values[d].shape().DebugString()
                << ").";
          }
          VLOG(5) << "Dense output tensor for key '"
                  << dataset()->dense_keys_[d] << "' is "
                  << avro_result.dense_values[d].SummarizeValue(3);
          (*out_tensors)[output_index] = avro_result.dense_values[d];
        }
        for (int d = 0; d < dataset()->sparse_keys_.size(); ++d) {
          int output_index =
              dataset()->key_to_output_index_.at(dataset()->sparse_keys_[d]);
          (*out_tensors)[output_index] =
              Tensor(ctx->allocator({}), DT_VARIANT, {3});
          Tensor& serialized_sparse = (*out_tensors)[output_index];
          auto serialized_sparse_t = serialized_sparse.vec<Variant>();
          serialized_sparse_t(0) = avro_result.sparse_indices[d];
          serialized_sparse_t(1) = avro_result.sparse_values[d];
          serialized_sparse_t(2) = avro_result.sparse_shapes[d];
          CHECK(serialized_sparse.dtype() ==
                dataset()->output_dtypes()[output_index])
              << "Got wrong type for key '" << dataset()->sparse_keys_[d] << "'"
              << " (expected "
              << DataTypeString(dataset()->output_dtypes()[output_index])
              << ", got " << DataTypeString(serialized_sparse.dtype()) << ").";
          CHECK(dataset()->output_shapes_[output_index].IsCompatibleWith(
              serialized_sparse.shape()))
              << "Got wrong shape for key '" << dataset()->sparse_keys_[d]
              << "'"
              << " (expected "
              << dataset()->output_shapes_[output_index].DebugString()
              << ", got " << serialized_sparse.shape().DebugString() << ").";

          VLOG(5) << "Sparse output tensor for " << dataset()->sparse_keys_[d]
                  << " has ";
          VLOG(5) << "Indices "
                  << avro_result.sparse_indices[d].SummarizeValue(3);
          VLOG(5) << "Values "
                  << avro_result.sparse_values[d].SummarizeValue(3);
          VLOG(5) << "Shapes "
                  << avro_result.sparse_shapes[d].SummarizeValue(3);
        }

        return Status::OK();
      }

      mutex mu_;
      size_t current_file_index_ TF_GUARDED_BY(mu_) = 0;
      std::unique_ptr<AvroFileStreamReader> reader_ TF_GUARDED_BY(mu_);
    };  // class Iterator

    static AvroParseConfig BuildConfig(
        int64 batch_size, bool drop_remainder, size_t input_stream_buffer_size,
        size_t avro_data_buffer_size, const std::vector<string>& dense_keys,
        const DataTypeVector& dense_types,
        const std::vector<PartialTensorShape>& dense_shapes,
        const std::vector<Tensor>& dense_defaults,
        const std::vector<string>& sparse_keys,
        const DataTypeVector& sparse_types) {
      AvroParseConfig config;
      // Create the config
      config.batch_size = batch_size;
      config.drop_remainder = drop_remainder;
      config.input_stream_buffer_size = input_stream_buffer_size;
      config.avro_data_buffer_size = avro_data_buffer_size;

      for (int d = 0; d < dense_keys.size(); ++d) {
        // If the user did not provide shape information and the first dimension
        // is -1 for the batch
        bool variable_length =
            dense_shapes[d].dims() == 1 && dense_shapes[d].dim_size(0) == -1;
        config.dense.push_back({dense_keys[d], dense_types[d], dense_shapes[d],
                                dense_defaults[d], variable_length});
      }
      for (int d = 0; d < sparse_keys.size(); ++d) {
        config.sparse.push_back({sparse_keys[d], sparse_types[d]});
      }

      return config;
    }

    const std::vector<string> filenames_;
    const string reader_schema_;
    const std::vector<Tensor> dense_defaults_;
    const std::vector<string> sparse_keys_;
    const std::vector<string> dense_keys_;
    const std::map<string, int> key_to_output_index_;
    const DataTypeVector sparse_types_;
    const DataTypeVector dense_types_;
    const std::vector<PartialTensorShape> dense_shapes_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const AvroParseConfig config_;
  };  // class Dataset

  const int graph_def_version_;
  std::string reader_schema_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::vector<string> sparse_keys_;
  std::vector<string> dense_keys_;
  DataTypeVector sparse_types_;
  DataTypeVector dense_types_;
  std::vector<PartialTensorShape> dense_shapes_;
};  // class AvroDatasetOp

// Register the kernel implementation for AvroDataset.
REGISTER_KERNEL_BUILDER(Name("IO>AvroDataset").Device(DEVICE_CPU),
                        AvroDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
