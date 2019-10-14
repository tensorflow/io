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

#ifndef TENSORFLOW_IO_AZURE_AZFS_RANDOM_ACCESS_FILE_H
#define TENSORFLOW_IO_AZURE_AZFS_RANDOM_ACCESS_FILE_H

#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

class AzBlobRandomAccessFile : public RandomAccessFile {
 public:
  AzBlobRandomAccessFile(const std::string& account,
                         const std::string& container,
                         const std::string& object);
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override;

 private:
  std::string account_;
  std::string container_;
  std::string object_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_IO_AZURE_AZFS_RANDOM_ACCESS_FILE_H