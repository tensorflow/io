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

#include "api/Compiler.hh"
#include "api/Decoder.hh"
#include "api/Generic.hh"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow_io/core/kernels/avro/utils/avro_parser_tree.h"

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
          variable_length(variable_length) {}
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
        : feature_name(feature_name), dtype(dtype) {}
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

int ResolveDefaultShape(TensorShape* resolved,
                        const PartialTensorShape& default_shape,
                        int64 batch_size) {
  // If default is not given or a scalar, do not resolve nor replicate
  if (default_shape.dims() < 1 ||
      (default_shape.dims() == 1 && default_shape.dim_size(0) <= 1)) {
    return 0;
  }

  // TODO: Check that dense shape matches dimensions, handle cases where the
  // default is not given
  PartialTensorShape full_shape(
      PartialTensorShape({batch_size}).Concatenate(default_shape));
  return full_shape.AsTensorShape(resolved);
}

std::map<string, Tensor> CreateTensorDefaults(const AvroParserConfig& config) {
  std::map<string, Tensor> defaults;
  for (const AvroParserConfig::Dense& dense : config.dense) {
    defaults[dense.feature_name] = dense.default_value;
  }
  return defaults;
}

class StringDatumRangeReader {
 public:
  StringDatumRangeReader(const gtl::ArraySlice<tstring>& serialized,
                         size_t start, size_t end)
      : serialized_(serialized),
        current_(start),
        end_(end),
        decoder_(avro::binaryDecoder()) {}

  bool read(avro::GenericDatum& datum) {
    if (current_ < end_) {
      std::unique_ptr<avro::InputStream> in =
          avro::memoryInputStream((const uint8_t*)serialized_[current_].data(),
                                  serialized_[current_].length());
      decoder_->init(*in);
      avro::GenericReader::read(*decoder_, datum);
      current_++;
      return true;
    }
    return false;
  }

 private:
  const gtl::ArraySlice<tstring>& serialized_;
  size_t current_;
  const size_t end_;
  avro::DecoderPtr decoder_;
};

// Borrowed most code/concepts from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/example_proto_fast_parsing.cc

// Preserves the order of parsed items
Status ParseAvro(const AvroParserConfig& config,
                 const AvroParserTree& parser_tree,
                 const avro::ValidSchema& reader_schema,
                 const gtl::ArraySlice<tstring>& serialized,
                 thread::ThreadPool* thread_pool, AvroResult* result) {
  DCHECK(result != nullptr);
  using clock = std::chrono::system_clock;
  using ms = std::chrono::duration<double, std::milli>;
  const auto before = clock::now();
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

  // avro_num_minibatches_ is int64 in the op interface. If not set
  // the default value is 0.
  size_t avro_num_minibatches_;

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
    if (avro_num_minibatches_) {
      VLOG(5) << "Overriding num_minibatches with " << avro_num_minibatches_;
      result = avro_num_minibatches_;
    }
    // This is to ensure users can control the num minibatches all the way down
    // to size of 1(no parallelism).
    const size_t min_minibatches = std::min<size_t>(1, serialized.size());
    const size_t max_minibatches = 64;
    return std::max<size_t>(min_minibatches,
                            std::min<size_t>(max_minibatches, result));
  }();

  auto first_of_minibatch = [&](size_t minibatch) -> size_t {
    return (serialized.size() * minibatch) / num_minibatches;
  };

  VLOG(5) << "Computed " << num_minibatches << " minibatches";

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

  // Note, using vector here is thread safe since all operations inside the
  // multi-threaded region for a vector are thread safe
  std::vector<std::map<string, ValueStoreUniquePtr>> buffers(num_minibatches);

  std::vector<Status> status_of_minibatch(num_minibatches);

  const std::map<string, Tensor>& defaults = CreateTensorDefaults(config);

  auto ProcessMiniBatch = [&](size_t minibatch) {
    size_t start = first_of_minibatch(minibatch);
    size_t end = first_of_minibatch(minibatch + 1);
    StringDatumRangeReader range_reader(serialized, start, end);
    auto read_value = [&](avro::GenericDatum& d) {
      return range_reader.read(d);
    };
    VLOG(5) << "Processing minibatch " << minibatch;
    status_of_minibatch[minibatch] = parser_tree.ParseValues(
        &buffers[minibatch], read_value, reader_schema, defaults);
  };
  const auto before_parse = clock::now();
  ParallelFor(ProcessMiniBatch, num_minibatches, thread_pool);
  const auto after_parse = clock::now();
  const ms parse_read_duration = after_parse - before_parse;
  VLOG(5) << "PARSER_TIMING: Time spend reading and parsing "
          << parse_read_duration.count() << " ms ";
  for (Status& status : status_of_minibatch) {
    TF_RETURN_IF_ERROR(status);
  }

  result->sparse_indices.reserve(config.sparse.size());
  result->sparse_values.reserve(config.sparse.size());
  result->sparse_shapes.reserve(config.sparse.size());
  result->dense_values.reserve(config.dense.size());

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
    result->sparse_values.emplace_back(sparse.dtype, value_shape);
    Tensor* sparse_tensor_values = &result->sparse_values.back();
    TensorShape index_shape;
    TF_RETURN_IF_ERROR((*value_store).GetSparseIndexShape(&index_shape));
    result->sparse_indices.emplace_back(DT_INT64, index_shape);
    Tensor* sparse_tensor_indices = &result->sparse_indices.back();
    TF_RETURN_IF_ERROR(
        (*value_store).MakeSparse(sparse_tensor_values, sparse_tensor_indices));

    int64 rank = result->sparse_indices[i_sparse].dim_size(
        1);  // rank is the 2nd dimension of the index
    result->sparse_shapes.emplace_back(DT_INT64, TensorShape({rank}));
    Tensor* sparse_tensor_shapes = &result->sparse_shapes.back();
    TF_RETURN_IF_ERROR(
        (*value_store).GetDenseShapeForSparse(sparse_tensor_shapes));

    VLOG(5) << "Sparse values: "
            << result->sparse_values[i_sparse].SummarizeValue(15);
    VLOG(5) << "Sparse indices: "
            << result->sparse_indices[i_sparse].SummarizeValue(15);
    VLOG(5) << "Sparse dense shapes: "
            << result->sparse_shapes[i_sparse].SummarizeValue(15);
    VLOG(5) << "Value shape: " << value_shape;
    VLOG(5) << "Index shape: " << index_shape;
    VLOG(5) << "Sparse dense shapes shape: "
            << result->sparse_shapes[i_sparse].shape();

    return Status::OK();
  };

  auto MergeDenseMinibatches = [&](size_t i_dense) -> Status {
    // TODO(fraudies): Fixed allocation for fixed length
    // if (!config.dense[d].variable_length) return;
    const AvroParserConfig::Dense& dense = config.dense[i_dense];
    const string& feature_name = dense.feature_name;

    VLOG(5) << "Working on feature: '" << feature_name << "'";

    std::vector<ValueStoreUniquePtr> values(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      values[i] = std::move(buffers[i][feature_name]);
      VLOG(5) << "Value " << i << ": " << (*values[i]).ToString(10);
    }

    VLOG(5) << "Merge for dense type: " << DataTypeString(dense.dtype);

    ValueStoreUniquePtr value_store;
    TF_RETURN_IF_ERROR(MergeAs(value_store, values, dense.dtype));

    VLOG(5) << "Merged value store: " << value_store->ToString(10);

    size_t batch_size = serialized.size();
    TensorShape default_shape;
    Tensor default_value;
    // If we can resolve the dense shape add batch, otherwise keep things as
    // they are
    if (ResolveDefaultShape(&default_shape, dense.default_value.shape(),
                            batch_size)) {
      default_value = Tensor(dense.dtype, default_shape);
      TF_RETURN_IF_ERROR(
          tensor::Concat(std::vector<Tensor>(batch_size, dense.default_value),
                         &default_value));
    } else {
      default_value = dense.default_value;
      default_shape = default_value.shape();
    }

    VLOG(5) << "Dense shape is " << dense.shape;
    VLOG(5) << "Default shape is " << default_shape;
    VLOG(5) << "Default value is " << default_value.SummarizeValue(9);

    TensorShape resolved_shape;
    TF_RETURN_IF_ERROR(
        (*value_store)
            .ResolveDenseShapeWithBatch(&resolved_shape, dense.shape,
                                        default_shape, batch_size));

    VLOG(5) << "Creating dense tensor for resolved shape: " << resolved_shape
            << " given the user shape " << dense.shape;

    result->dense_values.emplace_back(dense.dtype, resolved_shape);
    Tensor* dense_tensor = &result->dense_values.back();

    TF_RETURN_IF_ERROR(
        (*value_store).MakeDense(dense_tensor, resolved_shape, default_value));

    VLOG(5) << "Dense tensor " << dense_tensor->SummarizeValue(9);

    return Status::OK();
  };
  const auto before_sparse_merge = clock::now();
  for (size_t d = 0; d < config.sparse.size(); ++d) {
    TF_RETURN_IF_ERROR(MergeSparseMinibatches(d));
  }
  const auto after_sparse_merge = clock::now();
  const ms s_merge_duration = after_sparse_merge - before_sparse_merge;
  for (size_t d = 0; d < config.dense.size(); ++d) {
    TF_RETURN_IF_ERROR(MergeDenseMinibatches(d));
  }
  const auto after_dense_merge = clock::now();
  const ms d_merge_duration = after_dense_merge - after_sparse_merge;
  VLOG(5) << "PARSER_TIMING: Sparse merge duration" << s_merge_duration.count()
          << " ms ";

  VLOG(5) << "PARSER_TIMING: Dense merge duration" << d_merge_duration.count()
          << " ms ";
  return Status::OK();
}

// Inspired from here
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/example_parsing_ops.cc
class ParseAvroOp : public OpKernel {
 public:
  explicit ParseAvroOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Note, most sanity checks about lengths are done in the op definition
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_types", &sparse_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_types", &dense_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_shapes", &dense_shapes_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("avro_num_minibatches", &avro_num_minibatches_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_keys", &sparse_keys_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_keys", &dense_keys_));

    num_sparse_ = sparse_keys_.size();
    num_dense_ = dense_keys_.size();

    variable_length_.reserve(dense_shapes_.size());
    for (size_t d = 0; d < dense_shapes_.size(); ++d) {
      variable_length_[d] =
          dense_shapes_[d].dims() > 1 && dense_shapes_[d].dim_size(0) == -1;
    }

    // Check that avro_num_minibatches is not negative
    OP_REQUIRES(ctx, avro_num_minibatches_ >= 0,
                errors::InvalidArgument("Need avro_num_minibatches >= 0, got ",
                                        avro_num_minibatches_));

    string reader_schema_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reader_schema", &reader_schema_str));

    string error;
    std::istringstream ss(reader_schema_str);
    if (!avro::compileJsonSchema(ss, reader_schema_, error)) {
      OP_REQUIRES_OK(ctx,
                     errors::InvalidArgument("Avro schema error: ", error));
    }

    OP_REQUIRES_OK(ctx,
                   AvroParserTree::Build(&parser_tree_, CreateKeysAndTypes()));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* serialized;
    OpInputList dense_defaults;

    // Grab the input list arguments.
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_defaults", &dense_defaults));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(serialized->shape()),
                errors::InvalidArgument(
                    "Expected serialized to be a vector, got shape: ",
                    serialized->shape().DebugString()));
    OP_REQUIRES(ctx, dense_defaults.size() == num_dense_,
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_defaults.size(), " vs. ", num_dense_));

    for (size_t d = 0; d < num_dense_; ++d) {
      const Tensor& def_value = dense_defaults[d];
      OP_REQUIRES(ctx, def_value.dtype() == dense_types_[d],
                  errors::InvalidArgument(
                      "For key '", dense_keys_[d], "' ", "dense_defaults[", d,
                      "].dtype() == ", DataTypeString(def_value.dtype()),
                      " != dense_types_[", d,
                      "] == ", DataTypeString(dense_types_[d])));
    }

    AvroParserConfig config;
    for (size_t d = 0; d < num_dense_; ++d) {
      VLOG(7) << "Dense: Creating parser key " << dense_keys_[d]
              << " with type " << DataTypeString(dense_types_[d]);

      config.dense.push_back({dense_keys_[d], dense_types_[d], dense_shapes_[d],
                              std::move(dense_defaults[d]),
                              variable_length_[d]});
    }
    for (size_t d = 0; d < num_sparse_; ++d) {
      VLOG(7) << "Sparse: Creating parser key " << sparse_keys_[d]
              << " with type " << DataTypeString(sparse_types_[d]);
      config.sparse.push_back({sparse_keys_[d], sparse_types_[d]});
    }

    auto serialized_t = serialized->flat<tstring>();
    gtl::ArraySlice<tstring> slice(serialized_t.data(), serialized_t.size());

    AvroResult result;
    OP_REQUIRES_OK(
        ctx, ParseAvro(config, parser_tree_, reader_schema_, slice,
                       ctx->device()->tensorflow_cpu_worker_threads()->workers,
                       &result));

    OpOutputList dense_values;
    OpOutputList sparse_indices;
    OpOutputList sparse_values;
    OpOutputList sparse_shapes;
    OP_REQUIRES_OK(ctx, ctx->output_list("dense_values", &dense_values));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_indices", &sparse_indices));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_values", &sparse_values));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_shapes", &sparse_shapes));
    for (size_t d = 0; d < num_dense_; ++d) {
      dense_values.set(d, result.dense_values[d]);
    }
    for (size_t d = 0; d < num_sparse_; ++d) {
      sparse_indices.set(d, result.sparse_indices[d]);
      sparse_values.set(d, result.sparse_values[d]);
      sparse_shapes.set(d, result.sparse_shapes[d]);
    }
  }

 protected:
  AvroParserTree parser_tree_;
  std::vector<DataType> sparse_types_;
  std::vector<DataType> dense_types_;
  std::vector<string> sparse_keys_;
  std::vector<string> dense_keys_;
  std::vector<PartialTensorShape> dense_shapes_;
  std::vector<bool> variable_length_;
  avro::ValidSchema reader_schema_;
  size_t num_dense_;
  size_t num_sparse_;
  int64 avro_num_minibatches_;

 private:
  std::vector<std::pair<string, DataType>> CreateKeysAndTypes() {
    std::vector<std::pair<string, DataType>> keys_and_types;
    for (size_t d = 0; d < num_sparse_; ++d) {
      keys_and_types.push_back({sparse_keys_[d], sparse_types_[d]});
    }
    for (size_t d = 0; d < num_dense_; ++d) {
      keys_and_types.push_back({dense_keys_[d], dense_types_[d]});
    }
    return keys_and_types;
  }
};

REGISTER_KERNEL_BUILDER(Name("IO>ParseAvro").Device(DEVICE_CPU), ParseAvroOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
