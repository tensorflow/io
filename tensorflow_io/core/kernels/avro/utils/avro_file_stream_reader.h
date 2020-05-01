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
#ifndef TENSORFLOW_DATA_AVRO_FILE_STREAM_READER_H_
#define TENSORFLOW_DATA_AVRO_FILE_STREAM_READER_H_

#include "api/DataFile.hh"
#include "api/Stream.hh"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow_io/core/kernels/avro/utils/avro_parser_tree.h"

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

  // The buffer size for the input stream in By
  int64 input_stream_buffer_size;

  // The buffer size for the avro data stream in By
  int64 avro_data_buffer_size;
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
//    1. streams data from a random access file by blocks
//    2. feeds stream into avro stream to decode blocks into avro generic datum
//    3. parses the avro generic datum into tensors
// Supports batching
class AvroFileStreamReader {
 public:
  AvroFileStreamReader(Env* env, const string& filename,
                       const string& reader_schema_str,
                       const AvroParseConfig& config)
      : env_(env),
        filename_(filename),
        reader_schema_str_(reader_schema_str),
        config_(config),
        file_(nullptr),
        input_stream_(nullptr),
        buffered_input_stream_(nullptr),
        reader_(nullptr),
        allocator_(tensorflow::cpu_allocator()) {}

  // Open file and stream into file
  Status OnWorkStartup();

  // Reads up to batch_size elements and parses them into tensors
  Status Read(AvroResult* result);

 private:
  // Checks that there are no duplicate keys in the sparse feature names and
  // dense feature names
  std::vector<std::pair<string, DataType>> CreateKeysAndTypesFromConfig();

  // Resolves the shape from defaults into a dense tensor shape
  static int ResolveDefaultShape(TensorShape* resolved,
                                 const PartialTensorShape& default_shape,
                                 int64 batch_size);

  // TensorFlow env for IO
  Env* env_;

  // The filename from which we loaded the data
  const string filename_;

  // Avro reader schema
  const string reader_schema_str_;

  // Avro parser configuration
  const AvroParseConfig config_;

  // The random access file that we use to load the data
  std::unique_ptr<RandomAccessFile> file_;

  // The random access file stream
  std::unique_ptr<io::RandomAccessInputStream> input_stream_;

  // The buffered input stream as performance optimization
  std::unique_ptr<io::BufferedInputStream> buffered_input_stream_;

  // Avro data file reader to read generic datum from file stream
  std::unique_ptr<avro::DataFileReader<avro::GenericDatum>> reader_;

  // Avro schema
  avro::ValidSchema reader_schema_;

  // The parser tree that this reader leverages to parse the data into tensors
  AvroParserTree avro_parser_tree_;

  // Cache allocator here to avoid lock contention in
  // `tensorflow::cpu_allocator()`
  Allocator* allocator_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_AVRO_FILE_STREAM_READER_H_
