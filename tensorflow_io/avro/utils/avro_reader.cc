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

#include "tensorflow_io/avro/utils/avro_reader.h"
#include "tensorflow/core/framework/tensor_util.h"

namespace tensorflow {
namespace data {

Status AvroReader::OnWorkStartup() {

  // Allocate memory for the file part
  data_.reset(new (std::nothrow) char[file_size_]);
  if (data_.get() == nullptr) {
    return Status(errors::InvalidArgument("Unable to allocate ", file_size_/1024,
                                          " kB on memory in avro reader."));
  }

  // Read the file into the memory file
  StringPiece result;
  TF_RETURN_IF_ERROR((*file_).Read(0, file_size_, &result, data_.get()));

  bool do_resolve;
  TF_RETURN_IF_ERROR(AvroResolvedMemReader::DoResolve(&do_resolve, data_, file_size_,
    reader_schema_, filename_));

  // Create the memory reader
  if (do_resolve) {
    AvroResolvedMemReader* resolved_reader = new AvroResolvedMemReader();
    TF_RETURN_IF_ERROR(AvroResolvedMemReader::Create(resolved_reader, data_, file_size_,
      reader_schema_, filename_));
    avro_mem_reader_.reset(resolved_reader);
  } else {
    avro_mem_reader_.reset(new AvroMemReader());
    TF_RETURN_IF_ERROR(AvroMemReader::Create(avro_mem_reader_.get(), data_, file_size_, filename_));
  }

  // Create the parser tree
  TF_RETURN_IF_ERROR(AvroParserTree::Build(&avro_parser_tree_,
    (*avro_mem_reader_).GetNamespace(), CreateKeysAndTypesFromConfig()));

  return Status::OK();
}


Status AvroReader::Read(AvroResult* result) {

  // TODO(fraudies): Use callback for performance optimization
  std::vector<AvroValueSharedPtr> values;
  TF_RETURN_IF_ERROR((*avro_mem_reader_).ReadBatch(&values, config_.batch_size));

  int64 batch_size = values.size();

  LOG(INFO) << "Batch with " << values.size() << " values";

  TF_RETURN_IF_ERROR(avro_parser_tree_.ParseValues(&key_to_value_, values));

  LOG(INFO) << "Done parsing values";

  // Get sparse tensors
  size_t n_sparse = config_.sparse.size();
  (*result).sparse_indices.resize(n_sparse);
  (*result).sparse_values.resize(n_sparse);
  (*result).sparse_shapes.resize(n_sparse);

  for (size_t i_sparse = 0; i_sparse < n_sparse; ++i_sparse) {
    const AvroParseConfig::Sparse& sparse = config_.sparse[i_sparse];
    const ValueStoreUniquePtr& value_store = key_to_value_[sparse.feature_name];

    LOG(INFO) << "Converting sparse feature " << sparse.feature_name;
    LOG(INFO) << "Contents of value store " << (*value_store).ToString(10);

    TensorShape value_shape;
    TF_RETURN_IF_ERROR((*value_store).GetSparseValueShape(&value_shape));
    (*result).sparse_values[i_sparse] = Tensor(allocator_, sparse.dtype, value_shape);

    TensorShape index_shape;
    TF_RETURN_IF_ERROR((*value_store).GetSparseIndexShape(&index_shape));
    (*result).sparse_indices[i_sparse] = Tensor(allocator_, DT_INT64, index_shape);

    TF_RETURN_IF_ERROR((*value_store).MakeSparse(
      &(*result).sparse_values[i_sparse],
      &(*result).sparse_indices[i_sparse]));

    LOG(INFO) << "Sparse values: " << (*result).sparse_values[i_sparse].SummarizeValue(15);
    LOG(INFO) << "Sparse indices: " << (*result).sparse_indices[i_sparse].SummarizeValue(15);
    LOG(INFO) << "Value shape: " << value_shape;
    LOG(INFO) << "Index shape: " << index_shape;

    TensorShape size_shape;
    size_shape.AddDim(index_shape.dims());
    (*result).sparse_shapes[i_sparse] = Tensor(allocator_, DT_INT64, size_shape);
    TF_RETURN_IF_ERROR(ShapeToTensor(&(*result).sparse_shapes[i_sparse], index_shape));
  }

  // Get dense tensors
  size_t n_dense = config_.dense.size();
  (*result).dense_values.resize(n_dense);

  for (size_t i_dense = 0; i_dense < n_dense; ++i_dense) {
    const AvroParseConfig::Dense& dense = config_.dense[i_dense];

    const ValueStoreUniquePtr& value_store = key_to_value_[dense.feature_name];

    TensorShape default_shape;
    Tensor default_value;
    // If we can resolve the dense shape add batch, otherwise keep things as they are
    if (ResolveDefaultShape(&default_shape, dense.default_value.shape(), batch_size)) {
      default_value = Tensor(allocator_, dense.dtype, default_shape);
      TF_RETURN_IF_ERROR(tensor::Concat(
        std::vector<Tensor>(batch_size, dense.default_value),
        &default_value));
    } else {
      default_value = dense.default_value;
      default_shape = default_value.shape();
    }

    LOG(INFO) << "Default value is " << default_value.SummarizeValue(9);

    TensorShape resolved_shape;
    TF_RETURN_IF_ERROR((*value_store).ResolveDenseShape(&resolved_shape, dense.shape,
      default_shape));

    (*result).dense_values[i_dense] = Tensor(allocator_, dense.dtype, resolved_shape);

    LOG(INFO) << "Creating dense tensor for '" << dense.feature_name << "' with " << resolved_shape << " and user shape " << dense.shape;

    LOG(INFO) << (*value_store).ToString(10);

    TF_RETURN_IF_ERROR((*value_store).MakeDense(&(*result).dense_values[i_dense],
      resolved_shape, default_value));

    LOG(INFO) << "Dense tensor " << (*result).dense_values[i_dense].SummarizeValue(9);
  }

  return Status::OK();
}

int AvroReader::ResolveDefaultShape(TensorShape* resolved, const PartialTensorShape& default_shape,
  int64 batch_size) {

  // If default is not given or a scalar, do not resolve nor replicate
  if (default_shape.dims() < 1 || (default_shape.dims() == 1 && default_shape.dim_size(0) <= 1)) {
    return 0;
  }

  // TODO: Check that dense shape matches dimensions, handle cases where the default is not given
  PartialTensorShape full_shape(PartialTensorShape({batch_size}).Concatenate(default_shape));
  return full_shape.AsTensorShape(resolved);
}

// Assumes tensor has been allocated appropriate space -- not checked
Status AvroReader::ShapeToTensor(Tensor* tensor, const TensorShape& shape) {
  auto tensor_flat = (*tensor).flat<int64>();
  size_t n_dim = shape.dims();
  for (size_t i_dim = 0; i_dim < n_dim; ++i_dim) {
    tensor_flat(i_dim) = shape.dim_size(i_dim);
  }
  return Status::OK();
}

std::vector<std::pair<string, DataType>> AvroReader::CreateKeysAndTypesFromConfig() {
  std::vector<std::pair<string, DataType>> keys_and_types;
  for (const AvroParseConfig::Sparse& sparse : config_.sparse) {
    keys_and_types.push_back({sparse.feature_name, sparse.dtype});
  }
  for (const AvroParseConfig::Dense& dense : config_.dense) {
    keys_and_types.push_back({dense.feature_name, dense.dtype});
  }

  return keys_and_types;
}


}  // namespace data
}  // namespace tensorflow