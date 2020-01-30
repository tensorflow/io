/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#include <deque>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/blocking_counter.h"

#include "api/Generic.hh"
#include "api/Compiler.hh"
#include "api/Decoder.hh"

#include "tensorflow_io/core/utils/avro/avro_parser_tree.h"
#include "tensorflow_io/core/utils/avro/parallel_map_iterator.h"


namespace tensorflow {
namespace data {
namespace {


// Container for the parser configuration that holds
//    - dense tensor information (name, type, shape, default, variable length)
struct AvroParserConfig {

  // Parse configuration for dense tensors
  struct Dense {
    Dense(string feature_name, DataType dtype, PartialTensorShape shape,
          Tensor default_value, bool variable_length)
          : feature_name(feature_name),
            dtype(dtype),
            shape(std::move(shape)),
            default_value(std::move(default_value)),
            variable_length(variable_length) { }
    Dense() = default;

    // The feature name
    string feature_name;

    // The data type
    DataType dtype;

    // The partial input shape -- could be undefined
    PartialTensorShape shape;

    // The default tensor value
    Tensor default_value;

    // The user did not provide shape and we need to find the dimension
    bool variable_length;
  };

  // Parse configuration for sparse tensors
  struct Sparse {
    Sparse(string feature_name, DataType dtype)
        : feature_name(feature_name), dtype(dtype) { }
    Sparse() = default;

    // The feature name
    string feature_name;

    // The data type
    DataType dtype;
  };

  // A vector of dense configuration information
  std::vector<Dense> dense;

  // A vector of sparse configuration information
  std::vector<Sparse> sparse;
};


// Container for the
//    - sparse indices,
//    - sparse values,
//    - sparse shapes,
//    - dense values
struct AvroResult {
  std::vector<Tensor> sparse_indices;
  std::vector<Tensor> sparse_values;
  std::vector<Tensor> sparse_shapes;
  std::vector<Tensor> dense_values;
};


void ParallelFor(const std::function<void(size_t)>& f, size_t n,
                 thread::ThreadPool* thread_pool) {
  if (n == 0) return;
  if (thread_pool == nullptr) {
    for (size_t i = 0; i < n; ++i) {
      f(i);
    }
  } else {
    BlockingCounter counter(n - 1);
    for (size_t i = 1; i < n; ++i) {
      thread_pool->Schedule([i, &f, &counter] {
        f(i);
        counter.DecrementCount();
      });
    }
    f(0);
    counter.Wait();
  }
}

int ResolveDefaultShape(TensorShape* resolved, const PartialTensorShape& default_shape,
  int64 batch_size) {

  // If default is not given or a scalar, do not resolve nor replicate
  if (default_shape.dims() < 1 || (default_shape.dims() == 1 && default_shape.dim_size(0) <= 1)) {
    return 0;
  }

  // TODO: Check that dense shape matches dimensions, handle cases where the default is not given
  PartialTensorShape full_shape(PartialTensorShape({batch_size}).Concatenate(default_shape));
  return full_shape.AsTensorShape(resolved);
}

// Borrowed most code/concepts from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/example_proto_fast_parsing.cc
Status ParseAvro(const AvroParserConfig& config,
                 const AvroParserTree& parser_tree,
                 const avro::ValidSchema& reader_schema,
                 gtl::ArraySlice<string> serialized,
                 thread::ThreadPool* thread_pool,
                 AvroResult* result) {
  DCHECK(result != nullptr);

  // Allocate dense output for fixed length dense values
  // (variable-length dense and sparse and ragged have to be buffered).
/*  std::vector<Tensor> fixed_len_dense_values(config.dense.size());
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (config.dense[d].variable_length) continue;
    TensorShape out_shape;
    out_shape.AddDim(serialized.size());
    for (const int64 dim : config.dense[d].shape.dim_sizes()) {
      out_shape.AddDim(dim);
    }
    fixed_len_dense_values[d] = Tensor(config.dense[d].dtype, out_shape);
  }*/

  // This parameter affects performance in a big and data-dependent way.
  const size_t kMiniBatchSizeBytes = 50000;

  // Calculate number of minibatches.
  // In main regime make each minibatch around kMiniBatchSizeBytes bytes.
  // Apply 'special logic' below for small and big regimes.
  const size_t num_minibatches = [&] {
    size_t result = 0;
    size_t minibatch_bytes = 0;
    for (size_t i = 0; i < serialized.size(); i++) {
      if (minibatch_bytes == 0) {  // start minibatch
        result++;
      }
      minibatch_bytes += serialized[i].size() + 1;
      if (minibatch_bytes > kMiniBatchSizeBytes) {
        minibatch_bytes = 0;
      }
    }
    // 'special logic'
    const size_t min_minibatches = std::min<size_t>(8, serialized.size());
    const size_t max_minibatches = 64;
    return std::max<size_t>(min_minibatches,
                            std::min<size_t>(max_minibatches, result));
  }();

  auto first_of_minibatch = [&](size_t minibatch) -> size_t {
    return (serialized.size() * minibatch) / num_minibatches;
  };

  // TODO(lew): A big performance low-hanging fruit here is to improve
  //   num_minibatches calculation to take into account actual amount of work
  //   needed, as the size in bytes is not perfect. Linear combination of
  //   size in bytes and average number of features per example is promising.
  //   Even better: measure time instead of estimating, but this is too costly
  //   in small batches.
  //   Maybe accept outside parameter #num_minibatches?

  // Do minibatches in parallel.
  // TODO(fraudies): Convert dense tensor
  // TODO(fraudies): Might be faster to reformat inside the process minibatch
  // into vector
  std::vector<std::map<string, ValueStoreUniquePtr>> buffers(num_minibatches);
  std::vector<Status> status_of_minibatch(num_minibatches);
  auto ProcessMiniBatch = [&](size_t minibatch) {

    avro::GenericDatum datum(reader_schema);
    avro::DecoderPtr decoder = avro::binaryDecoder();

    size_t start = first_of_minibatch(minibatch);
    size_t end = first_of_minibatch(minibatch + 1);

    for (size_t e = start; e < end; ++e) {
      std::unique_ptr<avro::InputStream> in = avro::memoryInputStream(
        (const uint8_t*) serialized[e].data(), serialized[e].length());

      decoder->init(*in);
      //avro::decode(*decoder, datum); could not be found
      avro::GenericReader::read(*decoder, datum);
      status_of_minibatch[minibatch] =  parser_tree.ParseValue(
          &buffers[minibatch],
          datum);

      if (!status_of_minibatch[minibatch].ok()) break;
    }
  };

  ParallelFor(ProcessMiniBatch, num_minibatches, thread_pool);

  for (Status& status : status_of_minibatch) {
    TF_RETURN_IF_ERROR(status);
  }

  result->sparse_indices.reserve(config.sparse.size());
  result->sparse_values.reserve(config.sparse.size());
  result->sparse_shapes.reserve(config.sparse.size());
  result->dense_values.reserve(config.dense.size());

  auto MergeDenseMinibatches = [&](size_t i_dense) -> Status {
    // TODO(fraudies): Fixed allocation for fixed length
    // if (!config.dense[d].variable_length) return;
    const AvroParserConfig::Dense& dense = config.dense[i_dense];
    const string& feature_name = dense.feature_name;

    std::vector<ValueStoreUniquePtr> values(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        values[i] = std::move(buffers[i][feature_name]);
    }
    ValueStoreUniquePtr value_store;
    TF_RETURN_IF_ERROR(MergeAs(value_store, values, dense.dtype));

    size_t batch_size = serialized.size();

    TensorShape default_shape;
    Tensor default_value;
    // If we can resolve the dense shape add batch, otherwise keep things as they are
    if (ResolveDefaultShape(&default_shape, dense.default_value.shape(), batch_size)) {
      default_value = Tensor(dense.dtype, default_shape);
/*      TF_RETURN_IF_ERROR(tensor::Concat(
        std::vector<Tensor>(batch_size, dense.default_value),
        &default_value));*/
    } else {
      default_value = dense.default_value;
      default_shape = default_value.shape();
    }

    VLOG(5) << "Default shape is " << default_shape;
    VLOG(5) << "Default value is " << default_value.SummarizeValue(9);

    TensorShape resolved_shape;
    TF_RETURN_IF_ERROR((*value_store).ResolveDenseShape(&resolved_shape, dense.shape,
      default_shape));

    (*result).dense_values[i_dense] = Tensor(dense.dtype, resolved_shape);

    VLOG(5) << "Creating dense tensor for '" << dense.feature_name << "' with " << resolved_shape << " and user shape " << dense.shape;

    VLOG(5) << (*value_store).ToString(10);

    TF_RETURN_IF_ERROR((*value_store).MakeDense(&(*result).dense_values[i_dense],
      resolved_shape, default_value));

    VLOG(5) << "Dense tensor " << (*result).dense_values[i_dense].SummarizeValue(9);

    return Status::OK();
  };

  auto MergeSparseMinibatches = [&](size_t i_sparse) -> Status {
    const AvroParserConfig::Sparse& sparse = config.sparse[i_sparse];
    const string& feature_name = sparse.feature_name;

    std::vector<ValueStoreUniquePtr> values(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        values[i] = std::move(buffers[i][feature_name]);
    }
    ValueStoreUniquePtr value_store;
    TF_RETURN_IF_ERROR(MergeAs(value_store, values, sparse.dtype));

    VLOG(5) << "Converting sparse feature " << feature_name;
    VLOG(5) << "Contents of value store " << (*value_store).ToString(10);

    TensorShape value_shape;
    TF_RETURN_IF_ERROR((*value_store).GetSparseValueShape(&value_shape));
    (*result).sparse_values[i_sparse] = Tensor(sparse.dtype, value_shape);

    TensorShape index_shape;
    TF_RETURN_IF_ERROR((*value_store).GetSparseIndexShape(&index_shape));
    (*result).sparse_indices[i_sparse] = Tensor(DT_INT64, index_shape);

    TF_RETURN_IF_ERROR((*value_store).MakeSparse(
      &(*result).sparse_values[i_sparse],
      &(*result).sparse_indices[i_sparse]));

    int64 rank = (*result).sparse_indices[i_sparse].dim_size(1); // rank is the 2nd dimension of the index
    (*result).sparse_shapes[i_sparse] = Tensor(DT_INT64, TensorShape({rank}));
    TF_RETURN_IF_ERROR((*value_store).GetDenseShapeForSparse(&(*result).sparse_shapes[i_sparse]));

    VLOG(5) << "Sparse values: " << (*result).sparse_values[i_sparse].SummarizeValue(15);
    VLOG(5) << "Sparse indices: " << (*result).sparse_indices[i_sparse].SummarizeValue(15);
    VLOG(5) << "Sparse dense shapes: " << (*result).sparse_shapes[i_sparse].SummarizeValue(15);
    VLOG(5) << "Value shape: " << value_shape;
    VLOG(5) << "Index shape: " << index_shape;
    VLOG(5) << "Sparse dense shapes shape: " << (*result).sparse_shapes[i_sparse].shape();

    return Status::OK();
  };

  /*
  for (size_t d = 0; d < config.dense.size(); ++d) {
    result->dense_values.push_back(std::move(fixed_len_dense_values[d]));
  }
  */

  for (size_t d = 0; d < config.dense.size(); ++d) {
    TF_RETURN_IF_ERROR(MergeDenseMinibatches(d));
  }

  for (size_t d = 0; d < config.sparse.size(); ++d) {
    TF_RETURN_IF_ERROR(MergeSparseMinibatches(d));
  }

  return Status::OK();
}



// Mostly copied from here
// https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/core/kernels/data/experimental/parse_example_dataset_op.cc
// Changes are
// - Added reader schema
// - Removed handling of ragged tensors
// - Replaced FastParseExampleConfig by AvroParserConfig

class ParseAvroDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParseAvroDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {

    OP_REQUIRES_OK(ctx, ctx->input_list("dense_keys", &dense_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("sparse_keys", &sparse_keys));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("reader_schema", &reader_schema_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_types", &sparse_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_types", &dense_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_shapes", &dense_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sloppy", &sloppy_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 num_parallel_calls = 0;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                            &num_parallel_calls));
    OP_REQUIRES(
        ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutotune,
        errors::InvalidArgument(
            "num_parallel_calls must be greater than zero."));

    OpInputList dense_default_tensors;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("dense_defaults", &dense_default_tensors));

    OP_REQUIRES(ctx, dense_default_tensors.size() == dense_keys_.size(),
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_default_tensors.size(), " vs. ", dense_keys_.size()));

    std::vector<Tensor> dense_defaults(dense_default_tensors.begin(),
                                       dense_default_tensors.end());

    for (size_t d = 0; d < dense_keys_.size(); ++d) {
      const Tensor& def_value = dense_defaults[d];
      OP_REQUIRES(ctx, def_value.dtype() == dense_types_[d],
                  errors::InvalidArgument(
                      "dense_defaults[", d, "].dtype() == ",
                      DataTypeString(def_value.dtype()), " != dense_types_[", d,
                      "] == ", DataTypeString(dense_types_[d])));
    }

    std::map<string, int> key_to_output_index;
    for (size_t d = 0; d < dense_keys_.size(); ++d) {
      auto result = key_to_output_index.insert({dense_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          dense_keys_[d]));
    }
    for (size_t d = 0; d < sparse_keys_.size(); ++d) {
      auto result = key_to_output_index.insert({sparse_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          sparse_keys_[d]));
    }
    AvroParserConfig config(BuildConfig(dense_keys_, dense_types_, dense_shapes_,
                                        dense_defaults, sparse_keys_, sparse_types_));
    int i = 0;
    for (auto it = key_to_output_index.begin(); it != key_to_output_index.end();
         it++) {
      it->second = i++;
    }

    *output = new Dataset(ctx, input, reader_schema_, dense_defaults, sparse_keys_, dense_keys_,
                          std::move(key_to_output_index), std::move(config),
                          num_parallel_calls, sparse_types_, dense_types_,
                          dense_shapes_, output_types_, output_shapes_, sloppy_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const std::string& reader_schema,
            std::vector<Tensor> dense_defaults, std::vector<string> sparse_keys,
            std::vector<string> dense_keys,
            std::map<string, int> key_to_output_index,
            AvroParserConfig config,
            int32 num_parallel_calls,
            const DataTypeVector& sparse_types,
            const DataTypeVector& dense_types,
            const std::vector<PartialTensorShape>& dense_shapes,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            bool sloppy)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          reader_schema_(reader_schema),
          dense_defaults_(std::move(dense_defaults)),
          sparse_keys_(std::move(sparse_keys)),
          dense_keys_(std::move(dense_keys)),
          config_(std::move(config)),
          key_to_output_index_(std::move(key_to_output_index)),
          num_parallel_calls_(num_parallel_calls),
          sparse_types_(sparse_types),
          dense_types_(dense_types),
          dense_shapes_(dense_shapes),
          output_types_(output_types),
          output_shapes_(output_shapes),
          sloppy_(sloppy) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      std::unique_ptr<ParallelMapFunctor> parse_avro_functor =
          absl::make_unique<ParseAvroFunctor>(this);
      return NewParallelMapIterator(
          {this, strings::StrCat(prefix, "::ParseAvro")}, input_,
          std::move(parse_avro_functor), num_parallel_calls_, sloppy_,
          /*preserve_cardinality=*/true);
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "ParseAvroDatasetOp::Dataset";
    }

    int64 Cardinality() const override { return input_->Cardinality(); }

    // TODO(fraudies): Put me back when in TF 2.1
    // Status CheckExternalState() const override {
    //   return input_->CheckExternalState();
    // }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      Node* num_parallel_calls_node;
      std::vector<Node*> dense_defaults_nodes;
      dense_defaults_nodes.reserve(dense_defaults_.size());

      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallel_calls_node));

      for (const Tensor& dense_default : dense_defaults_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(dense_default, &node));
        dense_defaults_nodes.emplace_back(node);
      }

      AttrValue reader_schema_attr;
      AttrValue sparse_keys_attr;
      AttrValue dense_keys_attr;
      AttrValue sparse_types_attr;
      AttrValue dense_attr;
      AttrValue dense_shapes_attr;
      AttrValue sloppy_attr;

      b->BuildAttrValue(reader_schema_, &reader_schema_attr);
      b->BuildAttrValue(sparse_keys_, &sparse_keys_attr);
      b->BuildAttrValue(dense_keys_, &dense_keys_attr);
      b->BuildAttrValue(sparse_types_, &sparse_types_attr);
      b->BuildAttrValue(dense_types_, &dense_attr);
      b->BuildAttrValue(dense_shapes_, &dense_shapes_attr);
      b->BuildAttrValue(sloppy_, &sloppy_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(this,
                                       {
                                         {0, input_graph_node},
                                         {1, num_parallel_calls_node},
                                       },
                                       {{2, dense_defaults_nodes}},
                                       {{"reader_schema", reader_schema_attr},
                                        {"sparse_keys", sparse_keys_attr},
                                        {"dense_keys", dense_keys_attr},
                                        {"sparse_types", sparse_types_attr},
                                        {"Tdense", dense_attr},
                                        {"dense_shapes", dense_shapes_attr},
                                        {"sloppy", sloppy_attr}},
                                       output));
      return Status::OK();
    }

   private:
    class ParseAvroFunctor : public ParallelMapFunctor {
     public:
      explicit ParseAvroFunctor(const Dataset* dataset)
          : dataset_(dataset) {}

      void MapFunc(IteratorContext* ctx, const string& prefix,
                   std::vector<Tensor> input, std::vector<Tensor>* output,
                   StatusCallback callback) override {
        (*ctx->runner())([this, ctx, prefix, input, output, callback]() {
          thread::ThreadPool* device_threadpool =
              ctx->flr()->device()->tensorflow_cpu_worker_threads()->workers;
          std::vector<string> slice_vec;
          for (const Tensor& t : input) {
            auto serialized_t = t.flat<string>();
            gtl::ArraySlice<string> slice(serialized_t.data(),
                                           serialized_t.size());
            for (auto it = slice.begin(); it != slice.end(); it++)
              slice_vec.push_back(*it);
          }

          string error;
          avro::ValidSchema reader_schema;
          std::istringstream ss(dataset_->reader_schema_);
          if (!avro::compileJsonSchema(ss, reader_schema, error)) {
            return errors::InvalidArgument("Avro schema error: ", error);
          }

          // Handle namespace
          string avro_namespace(reader_schema.root()->hasName() ? reader_schema.root()->name().ns() : "");
          VLOG(3) << "Retrieved namespace" << avro_namespace;

          AvroParserTree parser_tree;
          TF_RETURN_IF_ERROR(AvroParserTree::Build(&parser_tree, avro_namespace,
            ParseAvroFunctor::CreateKeysAndTypes(dataset_->config_)));

          AvroResult avro_result;
          Status s = ParseAvro(dataset_->config_, parser_tree,
                               reader_schema, slice_vec,
                               device_threadpool, &avro_result);
          if (s.ok()) {
            (*output).resize(dataset_->key_to_output_index_.size());
            for (size_t d = 0; d < dataset_->dense_keys_.size(); ++d) {
              int output_index =
                  dataset_->key_to_output_index_.at(dataset_->dense_keys_[d]);
              CheckOutputTensor(avro_result.dense_values[d], d,
                                output_index);
              (*output)[output_index] = avro_result.dense_values[d];
            }
            for (size_t d = 0; d < dataset_->sparse_keys_.size(); ++d) {
              int output_index =
                  dataset_->key_to_output_index_.at(dataset_->sparse_keys_[d]);
              (*output)[output_index] =
                  Tensor(ctx->allocator({}), DT_VARIANT, {3});
              Tensor& serialized_sparse = (*output)[output_index];
              auto serialized_sparse_t = serialized_sparse.vec<Variant>();
              serialized_sparse_t(0) = avro_result.sparse_indices[d];
              serialized_sparse_t(1) = avro_result.sparse_values[d];
              serialized_sparse_t(2) = avro_result.sparse_shapes[d];
              CheckOutputTensor(serialized_sparse, d, output_index);
            }
          }
          callback(s);
        });
      }

     private:
      inline void CheckOutputTensor(const Tensor& tensor, size_t value_index,
                                    size_t output_index) const {
        DCHECK(tensor.dtype() == dataset_->output_dtypes()[output_index])
            << "Got wrong type for ParseAvro return value "
            << value_index << " (expected "
            << DataTypeString(dataset_->output_dtypes()[output_index])
            << ", got " << DataTypeString(tensor.dtype()) << ").";
        DCHECK(dataset_->output_shapes()[output_index].IsCompatibleWith(
            tensor.shape()))
            << "Got wrong shape for ParseAvro return value "
            << value_index << " (expected "
            << dataset_->output_shapes()[output_index].DebugString() << ", got "
            << tensor.shape().DebugString() << ").";
      }

      static std::vector<std::pair<string, DataType>> CreateKeysAndTypes(
        const AvroParserConfig& config) {

        std::vector<std::pair<string, DataType>> keys_and_types;
        for (const AvroParserConfig::Sparse& sparse : config.sparse) {
          keys_and_types.push_back({sparse.feature_name, sparse.dtype});
        }
        for (const AvroParserConfig::Dense& dense : config.dense) {
          keys_and_types.push_back({dense.feature_name, dense.dtype});
        }

        return keys_and_types;
      }

      const Dataset* dataset_;
    };

    const DatasetBase* const input_;
    const std::string reader_schema_;
    const std::vector<Tensor> dense_defaults_;
    const std::vector<string> sparse_keys_;
    const std::vector<string> dense_keys_;
    const AvroParserConfig config_;
    const std::map<string, int> key_to_output_index_;
    const int64 num_parallel_calls_;
    const DataTypeVector sparse_types_;
    const DataTypeVector dense_types_;
    const std::vector<PartialTensorShape> dense_shapes_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const bool sloppy_;
  };

  static AvroParserConfig BuildConfig(
    const std::vector<string>& dense_keys,
    const DataTypeVector& dense_types,
    const std::vector<PartialTensorShape>& dense_shapes,
    const std::vector<Tensor>& dense_defaults,
    const std::vector<string>& sparse_keys,
    const DataTypeVector& sparse_types) {

    AvroParserConfig config;
    for (size_t d = 0; d < dense_keys.size(); ++d) {
      // If the user did not provide shape information and the first dimension is -1 for the batch
      bool variable_length = dense_shapes[d].dims() > 1
                          && dense_shapes[d].dim_size(0) == -1;
      config.dense.push_back({dense_keys[d], dense_types[d], std::move(dense_shapes[d]),
                              std::move(dense_defaults[d]), variable_length});
    }
    for (size_t d = 0; d < sparse_keys.size(); ++d) {
      config.sparse.push_back({sparse_keys[d], sparse_types[d]});
    }
    return config;
  }

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  bool sloppy_;
  std::string reader_schema_;
  std::vector<string> sparse_keys_;
  std::vector<string> dense_keys_;
  DataTypeVector sparse_types_;
  DataTypeVector dense_types_;
  std::vector<PartialTensorShape> dense_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("IO>ParseAvro").Device(DEVICE_CPU),
                        ParseAvroDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow