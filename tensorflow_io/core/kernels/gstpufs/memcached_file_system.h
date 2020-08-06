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

#ifndef TENSORFLOW_IO_GSTPUFS_MEMCACHED_FILE_SYSTEM_H_
#define TENSORFLOW_IO_GSTPUFS_MEMCACHED_FILE_SYSTEM_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/cloud/gcs_file_system.h"
#include "tensorflow_io/core/kernels/gstpufs/gce_memcached_server_list_provider.h"
#include "tensorflow_io/core/kernels/gstpufs/memcached_file_block_cache.h"

namespace tensorflow {

// Environment variable that can be used to override the number of memcached
// clients that the file system keeps open for the distributed memcached.
constexpr char kMemcachedClientPoolSize[] = "MEMCACHED_CLIENT_POOL_SIZE";
// With 64 clients is enough for large workloads at ~128MB block size.
constexpr size_t kDefaultMemcachedClientPoolSize = 64;

// Google Cloud Storage implementation of a file system that contains a default
// block-cache that is useful for TPU platform reads of GCS data.
//
// The clients should use MemcachedGcsFileSystem defined below,
// which adds retry logic to GCS operations.
class MemcachedGcsFileSystem : public GcsFileSystem {
 public:
  // Main constructor used (via RetryingFileSystem) throughout Tensorflow.
  MemcachedGcsFileSystem();

 protected:
  std::unique_ptr<FileBlockCache> MakeFileBlockCache(
      size_t block_size, size_t max_bytes, uint64 max_staleness) override;

  // If the distributed cache is not specified for use in the env variables
  // the TPU GCS File System will simply be a wrapper on top of GCS File System
  // that changes no behavior in the file system.
  bool make_tpu_gcs_fs_cache_ = false;

  // Vector of pointers to the Memcached DAO objects, which is passed to the
  // distributed cached object has pointers to the clients.
  std::unique_ptr<std::vector<MemcachedDaoInterface*>> memcached_clients_;

 private:
  std::unique_ptr<GceMemcachedServerListProvider> server_list_provider_;
  TF_DISALLOW_COPY_AND_ASSIGN(MemcachedGcsFileSystem);

  // Owner of the Memcached DAO objects.
  std::unique_ptr<std::vector<std::unique_ptr<MemcachedDaoInterface>>>
      memcached_daos_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_IO_GSTPUFS_MEMCACHED_FILE_SYSTEM_H_
