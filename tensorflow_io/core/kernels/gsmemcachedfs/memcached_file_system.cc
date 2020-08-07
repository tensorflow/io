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

#include "tensorflow_io/core/kernels/gsmemcachedfs/memcached_file_system.h"

#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#ifdef _WIN32
#include <io.h>  // for _mktemp
#endif
#include "tensorflow/core/platform/cloud/ram_file_block_cache.h"

#ifdef _WIN32
#ifdef DeleteFile
#undef DeleteFile
#endif
#endif

namespace tensorflow {
namespace {

// The environment variable that overrides the type of cache to use.
constexpr char kClientCacheType[] = "GCS_CLIENT_CACHE_TYPE";
// Value of GCS_CLIENT_CACHE_TYPE, specifying None cache type.
constexpr char kNoneFileBlockCache[] = "None";
// Value of GCS_CLIENT_CACHE_TYPE, specifying replicated local RAM cache.
constexpr char kRamFileBlockCache[] = "RamFileBlockCache";
// Value of GCS_CLIENT_CACHE_TYPE, specifying distributed memcached cache.
constexpr char kMemcachedFileBlockCache[] = "MemcachedFileBlockCache";
// The environment variable that overrides the memcached server list to use.
constexpr char kMemcachedServerList[] = "GCS_MEMCACHED_SERVER_LIST";
// The environment variable that contains a list of memcached options.
constexpr char kMemcachedOptions[] = "GCS_MEMCACHED_OPTIONS";
// The environment variable that contains a list of GCS block fetcher options.
constexpr char kGcsBlockFetcherOptions[] = "GCS_BLOCK_FETCHER_OPTIONS";
// Number of gcs grpc call attempts.
constexpr char kMaxGrpcAttempts[] = "GCS_BLOCK_FETCHER_GRPC_MAX_ATTEMPTS";
// Sice of the local cached use by the distributed cache lient to cache blocks
// fetched from small read requests. Reads smaller than the block size causec a
// remote fetch for the entire block, thus we cache the entire block to be able
// to serve subsequent small reads from that block locally.
// 4GB local cache has shown very good hit-ratio and performance.
constexpr char kMemcachedLocalCachesize[] = "MEMCACHED_LOCAL_CACHE_SIZE_GB";

// How much time to initially wait before retrying a failed grpc.
constexpr absl::Duration kInitialGrpcRetry = absl::Seconds(1);
constexpr absl::Duration kMaxGrpcRetry = absl::Seconds(30);

/// \brief Utility function to split a comma delimited list of strings to an
/// ordered collection.
bool SplitByCommaToVector(StringPiece csv_list,
                          std::vector<string>* str_vector) {
  *str_vector = absl::StrSplit(csv_list, ',');
  return true;
}

bool StringPieceIdentity(StringPiece str, StringPiece* value) {
  *value = str;
  return true;
}

}  // namespace

MemcachedGcsFileSystem::MemcachedGcsFileSystem() : GcsFileSystem() {
  VLOG(1) << "Entering MemcachedGcsFileSystem::MemcachedGcsFileSystem";
  StringPiece client_cache_type;
  // We only run MEMCACHED GCS File System logic if we are set to use the
  // distributed cache. Otherwise skip MEMCACHED GCS FS and go straight to
  // the GCS file system.
  if (GetEnvVar(kClientCacheType, StringPieceIdentity, &client_cache_type) &&
      client_cache_type == kMemcachedFileBlockCache) {
    make_memcached_gcs_fs_cache_ = true;
    size_t block_size = kDefaultBlockSize;
    size_t max_bytes = kDefaultMaxCacheSize;
    uint64 max_staleness = kDefaultMaxStaleness;

    uint64 value;
    // Apply the overrides for the block size (MB), max bytes (MB), and max
    // staleness (seconds) if provided.
    if (GetEnvVar(kBlockSize, strings::safe_strtou64, &value)) {
      block_size = value * 1024 * 1024;
    }
    if (GetEnvVar(kMaxCacheSize, strings::safe_strtou64, &value)) {
      max_bytes = value * 1024 * 1024;
    }
    if (GetEnvVar(kMaxStaleness, strings::safe_strtou64, &value)) {
      max_staleness = value;
    }

    server_list_provider_ = absl::make_unique<GceMemcachedServerListProvider>(
        compute_engine_metadata_client_);

    VLOG(1) << "Reseting MEMCACHED-GCS cache with params: max_bytes = "
            << max_bytes << " ; "
            << "block_size = " << block_size << " ; "
            << "max_staleness = " << max_staleness;
    ResetFileBlockCache(block_size, max_bytes, max_staleness);
  }
}

// A helper function to build a FileBlockCache for MemcachedGcsFileSystem.
std::unique_ptr<FileBlockCache> MemcachedGcsFileSystem::MakeFileBlockCache(
    size_t block_size, size_t max_bytes, uint64 max_staleness) {
  StringPiece client_cache_type;
  if (!GetEnvVar(kClientCacheType, StringPieceIdentity, &client_cache_type) ||
      !make_memcached_gcs_fs_cache_) {
    // If no cache type is specified, the RamFileBlockCache will be used
    // by default.
    client_cache_type = kRamFileBlockCache;
  }
  VLOG(1) << "client_cache_type = " << client_cache_type;
  auto block_fetcher = [this](const string& fname, size_t offset, size_t n,
                              char* buffer, size_t* bytes_transferred) {
    return LoadBufferFromGCS(fname, offset, n, buffer, bytes_transferred);
  };
  VLOG(1) << "Creating " << client_cache_type;
  if (client_cache_type == kMemcachedFileBlockCache) {
    size_t client_pool_size = kDefaultMemcachedClientPoolSize;
    uint64 value;
    if (GetEnvVar(kMemcachedClientPoolSize, strings::safe_strtou64, &value)) {
      client_pool_size = value;
    }
    VLOG(1) << "Memcached client pool with " << client_pool_size << " clients.";
    memcached_clients_ =
        absl::make_unique<std::vector<MemcachedDaoInterface*>>();
    memcached_daos_ = absl::make_unique<
        std::vector<std::unique_ptr<MemcachedDaoInterface>>>();
    for (int i = 0; i < client_pool_size; ++i) {
      memcached_daos_->emplace_back(absl::make_unique<MemcachedDao>());
      memcached_clients_->push_back((*memcached_daos_)[i].get());
    }

    std::vector<string> servers;
    Status status = server_list_provider_->GetServerList(&servers);
    if (!status.ok()) {
      LOG(ERROR) << "Not able to get memcached server list. Will use a RAM "
                    "cache instead";
      std::unique_ptr<FileBlockCache> file_block_cache(new RamFileBlockCache(
          block_size, max_bytes, max_staleness, block_fetcher));
      return file_block_cache;
    }
    std::vector<string> options;
    if (GetEnvVar(kMemcachedOptions, SplitByCommaToVector, &options)) {
      VLOG(1) << "Override of memcached options";
    }

    size_t local_cache_size = 0;
    if (GetEnvVar(kMemcachedLocalCachesize, strings::safe_strtou64, &value)) {
      local_cache_size = value * 1024 * 1024 * 1024;
      VLOG(1) << "Distributed cache client has mini-reads cache of size = "
              << local_cache_size;
    }

    std::unique_ptr<FileBlockCache> file_block_cache(
        new MemcachedFileBlockCache(*memcached_clients_, block_size, max_bytes,
                                    max_staleness, local_cache_size, servers,
                                    options, block_fetcher));
    return file_block_cache;
  }

  if (client_cache_type == kNoneFileBlockCache) {
    std::unique_ptr<FileBlockCache> file_block_cache(
        new RamFileBlockCache(0, 0, max_staleness, block_fetcher));
    return file_block_cache;
  }

  // At this point, if the client_cache_type is something other than
  // the RamFileBlockCache, it means that the user has explicitly requested
  // an unsupported type.  Warn and continue with the default.
  if (client_cache_type != kRamFileBlockCache) {
    LOG(WARNING) << kClientCacheType << " set to unknown value \""
                 << client_cache_type << "\"; defaulting to "
                 << kRamFileBlockCache;
  }
  std::unique_ptr<FileBlockCache> file_block_cache(new RamFileBlockCache(
      block_size, max_bytes, max_staleness, block_fetcher));
  return file_block_cache;
}

}  // namespace tensorflow
