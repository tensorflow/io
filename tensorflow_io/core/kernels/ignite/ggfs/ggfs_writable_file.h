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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_GGFS_GGFS_WRITABLE_FILE_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_GGFS_GGFS_WRITABLE_FILE_H_

#include "tensorflow/core/platform/file_system.h"
#include "tensorflow_io/core/kernels/ignite/ggfs/ggfs_client.h"

namespace tensorflow {

class GGFSWritableFile : public WritableFile {
 public:
  GGFSWritableFile(const string &file_name,
                   std::unique_ptr<GGFSClient> &&client);
  ~GGFSWritableFile() override;
  Status Append(StringPiece data) override;
  Status Close() override;
  Status Flush() override;
  Status Sync() override;

 private:
  const string file_name_;
  std::unique_ptr<GGFSClient> client_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_GGFS_GGFS_WRITABLE_FILE_H_
