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

#include "tensorflow_io/ignite/kernels/ggfs/ggfs_random_access_file.h"

namespace tensorflow {

GGFSRandomAccessFile::GGFSRandomAccessFile(const string &file_name, std::unique_ptr<GGFSClient> &&client)
	: file_name_(file_name),
	  client_(std::move(client)) {
    LOG(INFO) << "Call GGFSRandomAccessFile constructor [file_name = " << file_name_ << "]";
}

GGFSRandomAccessFile::~GGFSRandomAccessFile() {
  LOG(INFO) << "Call GGFSRandomAccessFile destructor [file_name = " << file_name_ << "]";
}

Status GGFSRandomAccessFile::Read(uint64 offset, size_t n, StringPiece *result,
                                  char *scratch) const {
  LOG(INFO) << "Call Read [file_name = " << file_name_ << ", offset = " << offset << "]";

  

  return errors::Unimplemented("Not implemented yet");
}

}  // namespace tensorflow
