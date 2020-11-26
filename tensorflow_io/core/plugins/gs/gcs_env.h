/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_IO_CORE_PLUGINS_GS_GCS_ENV_H_
#define TENSORFLOW_IO_CORE_PLUGINS_GS_GCS_ENV_H_

#include "inttypes.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace io {
namespace gs {

typedef struct GCSThread GCSThread;
typedef struct GCSThreadOptions {
  size_t stack_size;
  size_t guard_size;
  int numa_node;
} GCSThreadOptions;

std::string GCSGetTempFileName(const std::string& extension);
uint64_t GCSNowSeconds(void);
void GCSDefaultThreadOptions(GCSThreadOptions* options);
GCSThread* GCSStartThread(const GCSThreadOptions* options,
                          const char* thread_name, void (*work_func)(void*),
                          void* param);
void GCSJoinThread(GCSThread* thread);

}  // namespace gs
}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_PLUGINS_GS_GCS_ENV_H_
