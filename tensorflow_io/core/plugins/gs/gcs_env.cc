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

#if !defined(_MSC_VER)
#include <sys/time.h>
#include <time.h>
#else
#include <time.h>
#include <windows.h>

#include <chrono>
#endif

#include "tensorflow_io/core/plugins/gs/gcs_env.h"

namespace tensorflow {
namespace io {
namespace gs {

uint64_t GCSNowSeconds(void) { return 0; }

void GCSDefaultThreadOptions(GCSThreadOptions* options) {
  options->stack_size = 0;
  options->guard_size = 0;
  options->numa_node = -1;
}

char* GCSGetTempFileName(const char* extension_) { return nullptr; }

GCSThread* GCSStartThread(const GCSThreadOptions* options,
                          const char* thread_name, void (*work_func)(void*),
                          void* param) {
  return nullptr;
}

void GCSJoinThread(GCSThread* thread) {}

}  // namespace gs
}  // namespace io
}  // namespace tensorflow
