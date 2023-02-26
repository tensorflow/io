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

#ifndef TENSORFLOW_IO_GSMEMCACHEDFS_GS_MEMCACHED_FILE_SYSTEM_H_
#define TENSORFLOW_IO_GSMEMCACHEDFS_GS_MEMCACHED_FILE_SYSTEM_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/tsl/platform/retrying_file_system.h"
#include "tensorflow_io/core/kernels/gsmemcachedfs/gce_memcached_server_list_provider.h"
#include "tensorflow_io/core/kernels/gsmemcachedfs/memcached_file_system.h"

namespace tensorflow {

// GsMemcachedFileSystem is implemented simply to register "gsmemcached://" as a
// file system scheme. It is used to add some file system optimizations for
// MEMCACHED on GCS datasets.
class GsMemcachedFileSystem : public MemcachedGcsFileSystem {
 public:
  GsMemcachedFileSystem() : MemcachedGcsFileSystem() {}

 protected:
  /// \brief Splits a GCS path to a bucket and an object.
  ///
  /// For example, "gsmemcached://bucket-name/path/to/file.txt" gets split into
  /// "bucket-name" and "path/to/file.txt".
  /// If fname only contains the bucket and empty_object_ok = true, the returned
  /// object is empty.
  Status ParseGcsPath(StringPiece fname, bool empty_object_ok, string* bucket,
                      string* object) override;
};

/// Google Cloud Storage implementation of a file system with retry on failures.
class RetryingGsMemcachedFileSystem
    : public tsl::RetryingFileSystem<GsMemcachedFileSystem> {
 public:
  RetryingGsMemcachedFileSystem()
      : tsl::RetryingFileSystem<GsMemcachedFileSystem>(
            absl::make_unique<GsMemcachedFileSystem>(),
            tsl::RetryConfig(100000 /* init_delay_time_us */)) {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_IO_GSMEMCACHEDFS_GS_MEMCACHED_FILE_SYSTEM_H_
