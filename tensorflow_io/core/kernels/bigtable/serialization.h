/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include "google/cloud/bigtable/table.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace io {

// Bigtable only stores byte buffers as values - except for int64 the server
// side does not have any notion of types. Tensorflow, needs to store shorter
// integers, floats, doubles, so we needed to decide on how. We chose to
// follow what HBase does, since there is a path for migrating from HBase to
// Bigtable. HBase stores integers as big-endian and floats as IEEE754
// (also big-endian). Given that integer endianness does not always match
// float endianness, and the fact that there are architectures where it is
// neither little nor big (BE-32), implementing this properly is non-trivial.
// Ideally, we would use a library to do that. XDR matches what HBase does,
// but it is not easily available on Windows, so we decided to go with a
// hybrid approach. On Windows we assume that integer endianness matches float
// endianness and implement the deserialization ourselves and everywhere else
// we use XDR. For that reason we provide two implementations
Status PutCellValueInTensor(Tensor& tensor, size_t index, DataType cell_type,
                            google::cloud::bigtable::Cell const& cell);

}  // namespace io
}  // namespace tensorflow

#endif /* SERIALIZATION_H */