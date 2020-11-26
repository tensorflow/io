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

#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#if defined(_MSC_VER)
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <string>
#include <vector>

#include "tensorflow/c/logging.h"
#include "tensorflow_io/core/plugins/gs/gcs_env.h"

namespace tensorflow {
namespace io {
namespace gs {
namespace {
// Returns a unique number every time it is called.
int64_t UniqueId() {
  static absl::Mutex mu;
  static int64_t id = 0;
  absl::MutexLock l(&mu);
  return ++id;
}

static bool IsAbsolutePath(absl::string_view path) {
  return !path.empty() && path[0] == '/';
}

std::string JoinPath(std::initializer_list<absl::string_view> paths) {
  std::string result;

  for (absl::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = std::string(path);
      continue;
    }

    if (result[result.size() - 1] == '/') {
      if (IsAbsolutePath(path)) {
        absl::StrAppend(&result, path.substr(1));
      } else {
        absl::StrAppend(&result, path);
      }
    } else {
      if (IsAbsolutePath(path)) {
        absl::StrAppend(&result, path);
      } else {
        absl::StrAppend(&result, "/", path);
      }
    }
  }

  return result;
}

}  // namespace

uint64_t GCSNowSeconds(void) {
  // TODO: Either implement NowSeconds here, or have TensorFlow API exposed
  std::abort();
}

void GCSDefaultThreadOptions(GCSThreadOptions* options) {
  options->stack_size = 0;
  options->guard_size = 0;
  options->numa_node = -1;
}

std::string GCSGetTempFileName(const std::string& extension) {
#if defined(_MSC_VER)
  char temp_dir[_MAX_PATH];
  DWORD retval;
  retval = GetTempPath(_MAX_PATH, temp_dir);
  if (retval > _MAX_PATH || retval == 0) {
    TF_Log(TF_FATAL, "Cannot get the directory for temporary files.");
  }

  char temp_file_name[_MAX_PATH];
  retval = GetTempFileNameA(temp_dir, "", UniqueId(), temp_file_name);
  if (retval > _MAX_PATH || retval == 0) {
    TF_Log(TF_FATAL, "Cannot get a temporary file in: %s", temp_dir);
  }

  std::string full_tmp_file_name(temp_file_name);
  full_tmp_file_name.append(extension);
  return full_tmp_file_name;
#else
  for (const char* dir : std::vector<const char*>(
           {getenv("TEST_TMPDIR"), getenv("TMPDIR"), getenv("TMP"), "/tmp"})) {
    if (!dir || !dir[0]) {
      continue;
    }
    struct stat statbuf;
    if (!stat(dir, &statbuf) && S_ISDIR(statbuf.st_mode)) {
      // UniqueId is added here because mkstemps is not as thread safe as it
      // looks. https://github.com/tensorflow/tensorflow/issues/5804 shows
      // the problem.
      std::string tmp_filepath;
      int fd;
      if (extension.length()) {
        tmp_filepath =
            JoinPath({dir, absl::StrCat("tmp_file_tensorflow_", UniqueId(),
                                        "_XXXXXX.", extension)});
        fd = mkstemps(&tmp_filepath[0], extension.length() + 1);
      } else {
        tmp_filepath = JoinPath(
            {dir, absl::StrCat("tmp_file_tensorflow_", UniqueId(), "_XXXXXX")});
        fd = mkstemp(&tmp_filepath[0]);
      }
      if (fd < 0) {
        TF_Log(TF_FATAL, "Failed to create temp file.");
      } else {
        if (close(fd) < 0) {
          TF_Log(TF_ERROR, "close() failed: %s", strerror(errno));
        }
        return tmp_filepath;
      }
    }
  }
  TF_Log(TF_FATAL, "No temp directory found.");
  std::abort();
#endif
}

GCSThread* GCSStartThread(const GCSThreadOptions* options,
                          const char* thread_name, void (*work_func)(void*),
                          void* param) {
  // TODO: Either implement StartThread here, or have TensorFlow API exposed
  std::abort();
  return nullptr;
}

void GCSJoinThread(GCSThread* thread) {
  // TODO: Either implement JoinThread here, or have TensorFlow API exposed
  std::abort();
}

}  // namespace gs
}  // namespace io
}  // namespace tensorflow
