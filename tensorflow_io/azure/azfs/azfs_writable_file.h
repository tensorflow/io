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

#ifndef TENSORFLOW_IO_AZURE_AZFS_WRITABLE_FILE_H
#define TENSORFLOW_IO_AZURE_AZFS_WRITABLE_FILE_H

#include <fstream>
#include <string>

#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

class AzBlobWritableFile : public WritableFile {
public:
  AzBlobWritableFile(const std::string &account, const std::string &container,
                     const std::string &object);
  ~AzBlobWritableFile() override;
  Status Append(StringPiece data) override;
  Status Close() override;
  Status Flush() override;
  Status Sync() override;

private:
  Status SyncImpl();
  Status CheckWritable() const;
  std::string account_;
  std::string container_;
  std::string object_;
  std::string tmp_content_filename_;
  std::ofstream outfile_;
  bool sync_needed_; // whether there is buffered data that needs to be synced
};

} // namespace tensorflow

#endif // TENSORFLOW_IO_AZURE_AZFS_WRITABLE_FILE_H