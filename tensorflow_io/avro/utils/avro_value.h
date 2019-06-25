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
#ifndef TENSORFLOW_DATA_AVRO_VALUE_H_
#define TENSORFLOW_DATA_AVRO_VALUE_H_

#include <avro.h>
#include <memory>

// Define some avro helpers
namespace tensorflow {
namespace data {

using AvroFileReaderPtr = std::unique_ptr<avro_file_reader_t, void(*)(avro_file_reader_t*)>;

using AvroSchemaPtr = std::unique_ptr<struct avro_obj_t, void(*)(avro_schema_t)>;

using AvroInterfacePtr = std::unique_ptr<avro_value_iface_t, void(*)(avro_value_iface_t*)>;

using AvroValueUniquePtr = std::unique_ptr<avro_value_t, void(*)(avro_value_t*)>;

using AvroValueSharedPtr = std::shared_ptr<avro_value_t>;

static void AvroSchemaDestructor(avro_schema_t schema) {
  avro_schema_decref(schema);
}

static void AvroInterfaceDestructor(avro_value_iface_t* iface)  {
  avro_value_iface_decref(iface);
}

static void AvroValueDestructor(avro_value_t* value) {
  avro_value_decref(value);
  free(value);
}

}
}

#endif // TENSORFLOW_DATA_AVRO_VALUE_H_