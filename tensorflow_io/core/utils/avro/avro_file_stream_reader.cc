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

#include "tensorflow_io/core/utils/avro/avro_file_stream_reader.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "api/DataFile.hh"
#include "api/Generic.hh"
#include "api/Compiler.hh"
#include <sstream>

namespace {
class AvroDataInputStream : public avro::InputStream {
public:
  AvroDataInputStream(tensorflow::io::BufferedInputStream* s, size_t avro_data_buffer_size)
    : buffered_input_stream_(s), avro_data_buffer_size_(avro_data_buffer_size) { }
  bool next(const uint8_t** data, size_t* len) override {

    if (*len <= 0 || *len > avro_data_buffer_size_) {
      *len = avro_data_buffer_size_;
    }

    if (do_seek) {
      buffered_input_stream_->Seek(pos_);
      do_seek = false;
    }

    buffered_input_stream_->ReadNBytes(*len, &chunk_);

    *data = (const uint8_t*) chunk_.data();
    *len = chunk_.size();

    pos_ += *len;

    return (*len != 0);
  }
  void backup(size_t len) override {
    do_seek = true;
    pos_ -= len;
  }
  void skip(size_t len) override {
    do_seek = true;
    pos_ += len;
  }
  size_t byteCount() const override {
    return pos_;
  }
private:
  tensorflow::io::BufferedInputStream* buffered_input_stream_;
  const size_t avro_data_buffer_size_;
  tensorflow::string chunk_;
  size_t pos_ = 0;
  bool do_seek = false;
};

}

namespace tensorflow {
namespace data {

Status AvroFileStreamReader::OnWorkStartup() {

  TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(filename_, &file_));

  uint64 size = 0;
  TF_RETURN_IF_ERROR(env_->GetFileSize(filename_, &size));

  VLOG(3) << "Creating input stream from file '" << filename_ << "' with size " << size/1024 << " kB";

  input_stream_.reset(new io::RandomAccessInputStream(file_.get()));

  buffered_input_stream_.reset(new io::BufferedInputStream(input_stream_.get(),
    config_.input_stream_buffer_size));

  string error;
  std::istringstream ss(reader_schema_str_);
  if (!avro::compileJsonSchema(ss, reader_schema_, error)) {
    return errors::InvalidArgument("Avro schema error: ", error);
  }

  std::unique_ptr<avro::InputStream> stream(
    static_cast<avro::InputStream*>(new AvroDataInputStream(buffered_input_stream_.get(),
      config_.avro_data_buffer_size)));

  reader_.reset(new avro::DataFileReader<avro::GenericDatum>(
    std::move(stream), reader_schema_));

  // Get the namespace
  string avro_namespace(reader_schema_.root()->hasName() ? reader_schema_.root()->name().ns() : "");
  VLOG(3) << "Retrieved namespace" << avro_namespace;

  // Create the parser tree
  TF_RETURN_IF_ERROR(AvroParserTree::Build(&avro_parser_tree_,
    avro_namespace, CreateKeysAndTypesFromConfig()));

  return Status::OK();
}

Status AvroFileStreamReader::Read(AvroResult* result) {
  std::map<string, ValueStoreUniquePtr> key_to_value;

  auto read_value = [&](avro::GenericDatum& d) { return reader_->read(d); };
  uint64 batch_size = 0;
  TF_RETURN_IF_ERROR(avro_parser_tree_.ParseValues(&key_to_value, read_value,
    reader_schema_, config_.batch_size, &batch_size));
  VLOG(5) << "Read and parsed " << batch_size << " elements";

  // Drop reminder if batch size is not the requested batch size
  if (config_.drop_remainder && batch_size != config_.batch_size) {
    VLOG(5) << "Drop " << batch_size << " remaining items";
    return errors::OutOfRange("eof");
  }

  // Get sparse tensors
  size_t n_sparse = config_.sparse.size();
  (*result).sparse_indices.resize(n_sparse);
  (*result).sparse_values.resize(n_sparse);
  (*result).sparse_shapes.resize(n_sparse);

  for (size_t i_sparse = 0; i_sparse < n_sparse; ++i_sparse) {
    const AvroParseConfig::Sparse& sparse = config_.sparse[i_sparse];
    const ValueStoreUniquePtr& value_store = key_to_value[sparse.feature_name];

    VLOG(5) << "Converting sparse feature " << sparse.feature_name;
    VLOG(5) << "Contents of value store " << (*value_store).ToString(10);

    TensorShape value_shape;
    TF_RETURN_IF_ERROR((*value_store).GetSparseValueShape(&value_shape));
    (*result).sparse_values[i_sparse] = Tensor(allocator_, sparse.dtype, value_shape);

    TensorShape index_shape;
    TF_RETURN_IF_ERROR((*value_store).GetSparseIndexShape(&index_shape));
    (*result).sparse_indices[i_sparse] = Tensor(allocator_, DT_INT64, index_shape);

    TF_RETURN_IF_ERROR((*value_store).MakeSparse(
      &(*result).sparse_values[i_sparse],
      &(*result).sparse_indices[i_sparse]));

    int64 rank = (*result).sparse_indices[i_sparse].dim_size(1); // rank is the 2nd dimension of the index
    (*result).sparse_shapes[i_sparse] = Tensor(allocator_, DT_INT64, TensorShape({rank}));
    TF_RETURN_IF_ERROR((*value_store).GetDenseShapeForSparse(&(*result).sparse_shapes[i_sparse]));

    VLOG(5) << "Sparse values: " << (*result).sparse_values[i_sparse].SummarizeValue(15);
    VLOG(5) << "Sparse indices: " << (*result).sparse_indices[i_sparse].SummarizeValue(15);
    VLOG(5) << "Sparse dense shapes: " << (*result).sparse_shapes[i_sparse].SummarizeValue(15);
    VLOG(5) << "Value shape: " << value_shape;
    VLOG(5) << "Index shape: " << index_shape;
    VLOG(5) << "Sparse dense shapes shape: " << (*result).sparse_shapes[i_sparse].shape();
  }

  // Get dense tensors
  size_t n_dense = config_.dense.size();
  (*result).dense_values.resize(n_dense);

  for (size_t i_dense = 0; i_dense < n_dense; ++i_dense) {
    const AvroParseConfig::Dense& dense = config_.dense[i_dense];

    const ValueStoreUniquePtr& value_store = key_to_value[dense.feature_name];

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

    VLOG(5) << "Default shape is " << default_shape;
    VLOG(5) << "Default value is " << default_value.SummarizeValue(9);

    TensorShape resolved_shape;
    TF_RETURN_IF_ERROR((*value_store).ResolveDenseShape(&resolved_shape, dense.shape,
      default_shape));

    (*result).dense_values[i_dense] = Tensor(allocator_, dense.dtype, resolved_shape);

    VLOG(5) << "Creating dense tensor for '" << dense.feature_name << "' with " << resolved_shape << " and user shape " << dense.shape;

    VLOG(5) << (*value_store).ToString(10);

    TF_RETURN_IF_ERROR((*value_store).MakeDense(&(*result).dense_values[i_dense],
      resolved_shape, default_value));

    VLOG(5) << "Dense tensor " << (*result).dense_values[i_dense].SummarizeValue(9);
  }

  return Status::OK();
}

int AvroFileStreamReader::ResolveDefaultShape(TensorShape* resolved, const PartialTensorShape& default_shape,
  int64 batch_size) {

  // If default is not given or a scalar, do not resolve nor replicate
  if (default_shape.dims() < 1 || (default_shape.dims() == 1 && default_shape.dim_size(0) <= 1)) {
    return 0;
  }

  // TODO: Check that dense shape matches dimensions, handle cases where the default is not given
  PartialTensorShape full_shape(PartialTensorShape({batch_size}).Concatenate(default_shape));
  return full_shape.AsTensorShape(resolved);
}

std::vector<std::pair<string, DataType>> AvroFileStreamReader::CreateKeysAndTypesFromConfig() {
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
