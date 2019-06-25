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
#ifndef TENSORFLOW_DATA_AVRO_MEM_READER_H_
#define TENSORFLOW_DATA_AVRO_MEM_READER_H_

#include <avro.h>
#include "tensorflow_io/avro/utils/avro_value.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// Avro memory reader assumes that the memory block contains
// - header information for avro files
// - schema information about the data
// - an arbitrary amount of values that follow
// This reader uses the file reader on a memory mapped file
class AvroMemReader {
  public:
    // Constructor
    AvroMemReader();

    // Create an avro memory reader from the provided memory
    // Assumes that the reader is initialized
    // Provide the filename for error messaging
    static Status Create(AvroMemReader* reader, const std::unique_ptr<char[]>& mem_data,
      const uint64 mem_size, const string& filename);

    // Read the next value
    // Note, value is only valid as long as no ReadNext is called since the internal method
    // re-uses the same memory for the next read
    virtual Status ReadNext(AvroValueUniquePtr& value);

    // Read the next batch of values
    // TODO(fraudies): For performance optimization convert this into a callback pattern,
    // e.g. iterator
    virtual Status ReadBatch(std::vector<AvroValueSharedPtr>* values, int64 batch_size);

    // Returns the schema string for this avro reader
    string GetNamespace() const;

  protected:
    // Avro file reader destructor
    static void AvroFileReaderDestructor(avro_file_reader_t* reader) {
      CHECK_GE(avro_file_reader_close(*reader), 0);
      free(reader);
    }

    // Used in lock
    mutex mu_;

    // Avro file reader
    AvroFileReaderPtr file_reader_ GUARDED_BY(mu_); // will close the file

    // Avro interface for writer value
    AvroInterfacePtr writer_iface_ GUARDED_BY(mu_);

    // Avro schema presentation for writer schema
    AvroSchemaPtr writer_schema_;
};

// Avro resolved memory reader that supports schema resolution
class AvroResolvedMemReader : public AvroMemReader {
  public:
    // Constructor
    AvroResolvedMemReader();

    // Create a resolved avro memory reader
    // Assumes that the reader has been initialized
    // The filename is used for error reporting
    static Status Create(AvroResolvedMemReader* reader, const std::unique_ptr<char[]>& mem_data,
      const uint64 mem_size, const string& reader_schema_str,
      const string& filename);

    // Method that checks if we need to resolve between the schema provided by the memory region
    // and the reader schema string
    static Status DoResolve(bool* resolve, const std::unique_ptr<char[]>& mem_data,
      const uint64 mem_size, const string& reader_schema_str, const string& filename);

    // Read next value
    virtual Status ReadNext(AvroValueUniquePtr& value);

    // Read batch of values
    // TODO(fraudies): For performance optimization convert this into a callback pattern,
    // e.g. iterator
    virtual Status ReadBatch(std::vector<AvroValueSharedPtr>* values, int64 batch_size);
  protected:

    // Avro interface for the reader value
    AvroInterfacePtr reader_iface_ GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_AVRO_MEM_READER_H_