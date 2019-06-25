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
#include "tensorflow_io/avro/utils/avro_mem_reader.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace data {


AvroMemReader::AvroMemReader() :
  file_reader_(nullptr, AvroFileReaderDestructor),
  writer_iface_(nullptr, AvroInterfaceDestructor),
  writer_schema_(nullptr, AvroSchemaDestructor) { }

Status AvroMemReader::Create(AvroMemReader* reader, const std::unique_ptr<char[]>& mem_data,
  const uint64 mem_size, const string& filename) {

  // Acquire the lock in case reader is still used to read
  mutex_lock l(reader->mu_);

  // Clear any previous error messages
  avro_set_error("");

  // Open a memory mapped file
  FILE* file(fmemopen(static_cast<void*>(mem_data.get()), mem_size, "rb"));
  if (file == nullptr) {
    return errors::InvalidArgument("Unable to open file ", filename, " for memory.");
  }

  // Create an avro file reader with that file
  avro_file_reader_t* file_reader = new avro_file_reader_t; // use tmp not to clean up a partially created reader
  if (avro_file_reader_fp(file, filename.c_str(), 1, file_reader) != 0) {
    free(file_reader);
    return errors::InvalidArgument("Unable to open file ", filename,
                                          " in avro reader. ", avro_strerror());
  }
  reader->file_reader_.reset(file_reader);

  // Get the writer schema
  avro_schema_t writer_schema = avro_file_reader_get_writer_schema(*reader->file_reader_);
  if (writer_schema == nullptr) {
    return errors::InvalidArgument("Unable to retrieve schema from file ", filename);
  }
  reader->writer_schema_.reset(writer_schema);

  // Get the writer interface for that schema
  avro_value_iface_t* writer_iface = avro_generic_class_from_schema(reader->writer_schema_.get());
  if (writer_iface == nullptr) {
    free(writer_iface);
    // TODO(fraudies): Get a string representation of the schema, use avro_schema_to_json
    return errors::ResourceExhausted("Unable to create interface for schema");
  }
  reader->writer_iface_.reset(writer_iface);

  return Status::OK();
}

Status AvroMemReader::ReadNext(AvroValueUniquePtr& value) {
  mutex_lock l(mu_);

  avro_set_error("");
  avro_value_t* next_value = new avro_value_t;
  // Initialize the value for that schema
  if (avro_generic_value_new(writer_iface_.get(), next_value) != 0) {
    return errors::InvalidArgument(
        "Unable to create instance for generic class.");
  }

  int ret = avro_file_reader_read_value(*file_reader_, next_value);
  // TODO(fraudies): Issue:
  // When reading from a memory mapped file we get this error
  // `Error reading file: Incorrect sync bytes`
  // Instead of EOF
  // Need to check why this is happening when opening the file with fmemopen and not with
  // fopen
  /*
  if (ret == EOF) {
    return errors::OutOfRange("eof");
  }
  if (ret != 0) {
    return errors::InvalidArgument("Unable to read value due to: ", avro_strerror());
  }
  */
  value.reset(next_value);
  if (ret != 0) {
    return errors::OutOfRange("eof");
  }
  return Status::OK();
}

Status AvroMemReader::ReadBatch(std::vector<AvroValueSharedPtr>* values, int64 batch_size) {

  mutex_lock l(mu_);


  for (int64 i_batch = 0; i_batch < batch_size; ++i_batch) {
    AvroValueUniquePtr next_value(new avro_value_t, AvroValueDestructor);

    // Initialize the value for that schema
    if (avro_generic_value_new(writer_iface_.get(), next_value.get()) != 0) {
      return errors::InvalidArgument(
          "Unable to create instance for generic class.");
    }

    int ret = avro_file_reader_read_value(*file_reader_, next_value.get());

    // TODO(fraudies): When migrating to avro 1.8 c++ handle this differently by checking for
    // the proper return value, if we have an error other than eof forward that to the user
    // Here it is currently treated as eof -- because the end produces a return code other
    // than eof when using a memory mapped file

    // If we receive a read error at the first value, then return EOF
    if (ret != 0 && i_batch == 0) {
      return errors::OutOfRange("eof");
    }

    // If we did not get a valid value, then stop and return this batch
    if (ret != 0) {
      break;
    }

    (*values).push_back(std::move(next_value));
  }

  return Status::OK();
}

// Return empty string if the namespace does not exist
string AvroMemReader::GetNamespace() const {
  const char* name = avro_schema_namespace(writer_schema_.get());
  return (name != nullptr) ? string(name) : "";
}

AvroResolvedMemReader::AvroResolvedMemReader() :
  AvroMemReader(),
  reader_iface_(nullptr, AvroInterfaceDestructor) { }

// An example of resolved reading can be found in this test case test_avro_984.c
// We follow that here
Status AvroResolvedMemReader::Create(AvroResolvedMemReader* reader, const std::unique_ptr<char[]>& mem_data,
  const uint64 mem_size, const string& reader_schema_str, const string& filename) {

  // Acquire the lock in case reader is still used to read
  mutex_lock l(reader->mu_);

  // Create a reader schema for the user passed string
  avro_schema_t reader_schema_tmp;
  if (avro_schema_from_json_length(reader_schema_str.data(),
                                   reader_schema_str.length(),
                                   &reader_schema_tmp) != 0) {
    return errors::InvalidArgument(
        "The provided json schema is invalid. ", avro_strerror());
  }
  AvroSchemaPtr reader_schema(reader_schema_tmp, AvroSchemaDestructor);

  // Create reader class
  avro_value_iface_t* reader_iface(avro_generic_class_from_schema(reader_schema.get()));
  if (reader_iface == nullptr) {
    // TODO(fraudies): Print the schemas in the error message
    return Status(errors::ResourceExhausted("Unable to create interface for schema"));
  }
  reader->reader_iface_.reset(reader_iface);

  // Open a memory mapped file
  FILE* file = fmemopen(static_cast<void*>(mem_data.get()), mem_size, "rb");
  if (file == nullptr) {
    return errors::InvalidArgument("Unable to open file ", filename, " for memory.");
  }

  // Closes the file handle
  avro_file_reader_t* file_reader = new avro_file_reader_t; // use tmp not to clean up a partially created reader
  if (avro_file_reader_fp(file, filename.c_str(), 1, file_reader) != 0) {
    free(file_reader);
    return errors::InvalidArgument("Unable to open file ", filename,
                                   " in avro reader. ", avro_strerror());
  }
  reader->file_reader_.reset(file_reader);

  // Get the writer schema
  AvroSchemaPtr writer_schema(avro_file_reader_get_writer_schema(*reader->file_reader_),
             AvroSchemaDestructor);
  if (writer_schema.get() == nullptr) {
    return errors::InvalidArgument("Unable to retrieve schema from file ", filename);
  }

  // Get the writer interface and initialize the value for that interface
  avro_value_iface_t* writer_iface(avro_resolved_writer_new(writer_schema.get(), reader_schema.get()));
  if (writer_iface == nullptr) {
    free(writer_iface);
    // TODO(fraudies): Get a string representation of the schema, use avro_schema_to_json
    return errors::InvalidArgument("Schemas are incompatible. ", avro_strerror());
  }
  reader->writer_iface_.reset(writer_iface);

  return Status::OK();
}

Status AvroResolvedMemReader::ReadNext(AvroValueUniquePtr& value) {
  mutex_lock l(mu_);

  avro_value_t* reader_value = new avro_value_t;
  avro_value_t* writer_value = new avro_value_t;

  // Initialize value for reader class
  if (avro_generic_value_new(reader_iface_.get(), reader_value) != 0) {
    return errors::InvalidArgument("Unable to create instance for generic reader class.");
  }

  // Create instance for resolved writer class
  if (avro_resolved_writer_new_value(writer_iface_.get(), writer_value) != 0) {
    return errors::InvalidArgument("Unable to create resolved writer value.");
  }
  avro_resolved_writer_set_dest(writer_value, reader_value);

  int ret = avro_file_reader_read_value(*file_reader_, writer_value);
  if (ret != 0) {
    return errors::OutOfRange("eof");
  }
  // TODO(fraudies): Issue:
  // When reading from a memory mapped file we get this error
  // `Error reading file: Incorrect sync bytes`
  // Instead of EOF
  // Need to check why this is happening when opening the file with fmemopen and not with
  // fopen
  /*
  if (ret == EOF) {
    return errors::OutOfRange("eof");
  }
  if (ret != 0) {
    return errors::InvalidArgument("Unable to read value due to: ", avro_strerror());
  }
  */
  // Transfer ownership for reader value
  value.reset(reader_value);
  // Cleanup writer value
  avro_value_decref(writer_value);
  free(writer_value);

  return Status::OK();
}

Status AvroResolvedMemReader::ReadBatch(std::vector<AvroValueSharedPtr>* values, int64 batch_size) {

  mutex_lock l(mu_);

  for (int64 i_batch = 0; i_batch < batch_size; ++i_batch) {

    AvroValueUniquePtr reader_value(new avro_value_t, AvroValueDestructor);
    AvroValueUniquePtr writer_value(new avro_value_t, AvroValueDestructor);

    // Initialize value for reader class
    if (avro_generic_value_new(reader_iface_.get(), reader_value.get()) != 0) {
      return errors::InvalidArgument("Unable to create instance for generic reader class.");
    }

    if (avro_resolved_writer_new_value(writer_iface_.get(), writer_value.get()) != 0) {
      return errors::InvalidArgument("Unable to create resolved writer value.");
    }
    avro_resolved_writer_set_dest(writer_value.get(), reader_value.get());

    int ret = avro_file_reader_read_value(*file_reader_, writer_value.get());

    // TODO(fraudies): When migrating to avro 1.8 c++ handle this differently by checking for
    // the proper return value, if we have an error other than eof forward that to the user
    // Here it is currently treated as eof -- because the end produces a return code other
    // than eof when using a memory mapped file
    // if (ret == EOF) {
    //    return errors::OutOfRange("eof");
    // }
    // if (ret != 0) {
    //    return errors::InvalidArgument("Unable to read value due to: ", avro_strerror());
    // }

    // If we receive a read error at the first value, then return EOF
    if (ret != 0 && i_batch == 0) {
      return errors::OutOfRange("eof");
    }

    // If we did not get a valid value, then stop and return this batch
    if (ret != 0) {
      break;
    }

    (*values).push_back(std::move(reader_value));

  }
  return Status::OK();
}


Status AvroResolvedMemReader::DoResolve(bool* resolve, const std::unique_ptr<char[]>& mem_data,
  const uint64 mem_size, const string& reader_schema_str, const string& filename) {

  // No schema supplied => no schema resolution is necessary
  if (reader_schema_str.length() <= 0) {
    *resolve = false;
    return Status::OK();
  }

  // Open the file to get the writer schema
  FILE* file(fmemopen(static_cast<void*>(mem_data.get()), mem_size, "r"));
  if (file == nullptr) {
    return errors::InvalidArgument("Unable to open file ", filename, " for memory.");
  }

  // Open the avro file reader
  avro_file_reader_t* file_reader_tmp = new avro_file_reader_t;
  if (avro_file_reader_fp(file, filename.c_str(), 1, file_reader_tmp) != 0) {
    free(file_reader_tmp);
    return errors::InvalidArgument("Unable to open file ", filename,
                                   " in avro reader. ", avro_strerror());
  }
  AvroFileReaderPtr file_reader(file_reader_tmp, AvroFileReaderDestructor);

  // Get the writer schema
  AvroSchemaPtr writer_schema(avro_file_reader_get_writer_schema(*file_reader),
             AvroSchemaDestructor);
  if (writer_schema.get() == nullptr) {
    return errors::InvalidArgument("Unable to retrieve schema from file ", filename);
  }

  // Create a reader schema for the user passed string
  avro_schema_t reader_schema_tmp;
  if (avro_schema_from_json_length(reader_schema_str.data(),
                                   reader_schema_str.length(),
                                   &reader_schema_tmp) != 0) {
    return errors::InvalidArgument("The provided json schema is invalid. ", avro_strerror());
  }
  AvroSchemaPtr reader_schema(reader_schema_tmp, AvroSchemaDestructor);

  // Do resolve only if the schemas are different
  *resolve = !avro_schema_equal(writer_schema.get(), reader_schema.get());

  return Status::OK();
}


}  // namespace data
}  // namespace tensorflow