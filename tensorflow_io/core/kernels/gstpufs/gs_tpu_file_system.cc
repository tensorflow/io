/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_io/core/kernels/gstpufs/gs_tpu_file_system.h"

namespace tensorflow {

Status GsTpuFileSystem::ParseGcsPath(StringPiece fname, bool empty_object_ok,
                                     string* bucket, string* object) {
  return ParseGcsPathForScheme(fname, "gstpu", empty_object_ok, bucket, object);
}

}  // namespace tensorflow

REGISTER_FILE_SYSTEM("gstpu", ::tensorflow::RetryingGsTpuFileSystem);
