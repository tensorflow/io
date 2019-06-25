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
#ifndef TENSORFLOW_DATA_AVRO_READER_H_
#define TENSORFLOW_DATA_AVRO_READER_H_

#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"

#include "tensorflow_io/avro/utils/avro_parser_tree.h"
#include "tensorflow_io/avro/utils/avro_mem_reader.h"

namespace tensorflow {
namespace data {

// Container for the parser configuration that holds
//    - dense tensor information (name, type, shape, default, variable length)
struct AvroParseConfig {

  // Parse configuration for dense tensors
  struct Dense {
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
    // The feature name
    string feature_name;

    // The data type
    DataType dtype;
  };

  // The size of the batch in number of elements
  int64 batch_size;

  // Whether we should drop the remainder of (n_data % batch_size) elements
  bool drop_remainder;

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


// The avro reader does the following
//    1. loads the data from the random access file into memory
//    2. Uses the avro memory reader to parse that data into avro values
//    3. Uses the avro parser tree to parse the avro values into tensors
// Supports batching
class AvroReader {
public:
  AvroReader(const std::unique_ptr<RandomAccessFile>& file, const uint64 file_size,
             const string& filename, const string& reader_schema, const AvroParseConfig& config)
    : file_(std::move(file)),
      file_size_(file_size),
      filename_(filename),
      reader_schema_(reader_schema),
      config_(config),
      allocator_(tensorflow::cpu_allocator()) { }

  // Call for startup to load data into memory, set up avro memory reader, and the parser tree.
  Status OnWorkStartup();

  // Reads up to batch_size elements and parses them into tensors
  Status Read(AvroResult* result);
private:

  // Assumes tensor has been allocated appropriate space -- not checked
  static Status ShapeToTensor(Tensor* tensor, const TensorShape& shape);

  // Checks that there are no duplicate keys in the sparse feature names and dense feature names
  std::vector<std::pair<string, DataType>> CreateKeysAndTypesFromConfig();

  // Resolves the shape from defaults into a dense tensor shape
  static int ResolveDefaultShape(TensorShape* resolved,
    const PartialTensorShape& default_shape,
    int64 batch_size);

  // The random access file that we use to load the data
  const std::unique_ptr<RandomAccessFile>& file_;

  // The file size from which we loaded the data
  const uint64 file_size_;

  // The filename from which we loaded the data
  const string filename_;

  // Avro reader schema
  const string reader_schema_;

  // Avro parser configuration
  const AvroParseConfig config_;

  // The memory reader
  std::unique_ptr<AvroMemReader> avro_mem_reader_;

  // The parser tree that this reader leverages to parse the data into tensors
  AvroParserTree avro_parser_tree_;

  // The data for this reader
  std::unique_ptr<char[]> data_;

  // Key to value mapping
  std::map<string, ValueStoreUniquePtr> key_to_value_;

  // Cache allocator here to avoid lock contention in `tensorflow::cpu_allocator()`
  Allocator* allocator_;
};

}
}

#endif  // TENSORFLOW_DATA_AVRO_READER_H_