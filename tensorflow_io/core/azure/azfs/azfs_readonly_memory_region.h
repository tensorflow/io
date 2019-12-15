/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_IO_AZURE_AZFS_READONLY_MEMORY_REGION_H
#define TENSORFLOW_IO_AZURE_AZFS_READONLY_MEMORY_REGION_H

#include <memory>

#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace io {

class AzBlobReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  AzBlobReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {}
  const void* data() override { return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override { return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_AZURE_AZFS_READONLY_MEMORY_REGION_H
