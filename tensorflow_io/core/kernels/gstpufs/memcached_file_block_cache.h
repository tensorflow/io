/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_IO_GSTPUFS_MEMCACHED_FILE_BLOCK_CACHE_H_
#define TENSORFLOW_IO_GSTPUFS_MEMCACHED_FILE_BLOCK_CACHE_H_

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include "tensorflow_io/core/kernels/gstpufs/memcached_dao_interface.h"

namespace tensorflow {

class MiniBlockCache {
 public:
  explicit MiniBlockCache(size_t max_size) : max_size_(max_size) {
    VLOG(1) << "MiniBlockCache max_size = " << max_size_;
  }

  // Add block to the cache.
  void Add(std::string key, size_t block_size, char* data)
      ABSL_LOCKS_EXCLUDED(mu_) {
    if (max_size_ == 0) {
      return;
    }
    mutex_lock lock(mu_);
    VLOG(3) << "MiniBlockCache Add: key = " << key
            << ", block_size = " << block_size
            << ", to current_size = " << keys_fifo_.size();
    if (!map_.contains(key)) {
      if (max_size_ < (size_ + block_size) && !keys_fifo_.empty()) {
        string pop_key = keys_fifo_.front();
        VLOG(3) << "MiniBlockCache pop key = " << pop_key;
        size_ -= map_[pop_key]->size();
        map_.erase(pop_key);
        keys_fifo_.pop();
      }
      keys_fifo_.push(key);
      map_[key] = absl::make_unique<std::vector<char>>();
    }
    map_[key]->assign(data, data + block_size);
    size_ += map_[key]->size();
  }

  // Peek map to check if the key is contained in it.
  bool Peek(std::string key) ABSL_LOCKS_EXCLUDED(mu_) {
    if (max_size_ == 0) {
      return false;
    }
    mutex_lock lock(mu_);
    return map_.contains(key);
  }

  // Get block from cache if it exists.
  bool Get(std::string key, int64 offset, size_t n, char* buffer,
           size_t* bytes_copied) ABSL_LOCKS_EXCLUDED(mu_) {
    if (max_size_ == 0) {
      *bytes_copied = 0;
      return false;
    }
    mutex_lock lock(mu_);
    if (!map_.contains(key) || offset > map_[key]->size()) {
      VLOG(3) << "MiniBlockCache MISS Get: key = " << key
              << ", offset = " << offset << ", n = " << n;
      *bytes_copied = 0;
      return false;
    }
    VLOG(3) << "MiniBlockCache HIT Get: key = " << key
            << ", offset = " << offset << ", n = " << n;

    int64 bytes_to_copy = n;
    if (offset + n > map_[key]->size()) {
      bytes_to_copy = map_[key]->size() - offset;
    }

    memcpy(buffer, map_[key]->data() + offset, bytes_to_copy);
    *bytes_copied = bytes_to_copy;
    return true;
  }

  // Mark block as FETCHING state if it is not fetching yet. If it was already
  // fetching then add thread to the list waiting on the block to be fetched.
  void Fetching(std::string key) ABSL_LOCKS_EXCLUDED(fetcher_mu_) {
    mutex_lock lock(fetcher_mu_);
    if (!fetching_map_.contains(key)) {
      fetching_map_[key] = std::make_shared<condition_variable>();
    } else {
      fetching_map_[key]->wait_for(lock, std::chrono::seconds(60));
    }
  }

  // Mark block as FETCHED and notify all threads waiting for it.
  void Fetched(std::string key) ABSL_LOCKS_EXCLUDED(fetcher_mu_) {
    mutex_lock lock(fetcher_mu_);
    if (fetching_map_.contains(key)) {
      fetching_map_[key]->notify_all();
    }
    fetching_map_.erase(key);
  }

 private:
  const size_t max_size_;
  mutable mutex mu_;
  size_t size_ ABSL_GUARDED_BY(mu_) = 0;
  std::queue<string> keys_fifo_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::unique_ptr<std::vector<char>>> map_
      ABSL_GUARDED_BY(mu_);
  mutable mutex fetcher_mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<condition_variable>>
      fetching_map_ ABSL_GUARDED_BY(fetcher_mu_);
};

// The callback executed when a block is not found in the cache, and needs to
// be fetched from the backing filesystem. This callback is provided when the
// cache is constructed. The returned Status should be OK as long as the
// read from the remote filesystem succeeded (similar to the semantics of the
// read(2) system call).
using BlockFetcher = std::function<Status(const string& filename, size_t offset,
                                          size_t buffer_size, char* buffer,
                                          size_t* bytes_transferred)>;

// Memcached Data Access Object class that wraps memcached. We should use this
// class for any access or configuration of memcached within the memcached file
// block cache. This construct is useful for testing, since it allows us to
// mock memcached in our unit tests.
class MemcachedDao : public MemcachedDaoInterface {
 public:
  MemcachedDao() {}

  explicit MemcachedDao(memcached_st* memcached_handle) {
    memcached_handle_ = memcached_handle;
  }

  memcached_st* MemcachedCreate() override {
    memcached_handle_ = memcached_create(nullptr);
    return memcached_handle_;
  }

  void MemcachedReset(memcached_st* memcached_handle) override {
    memcached_handle_ = memcached_handle;
  }

  memcached_return_t MemcachedBehaviorSet(const memcached_behavior_t flag,
                                          uint64_t data) override {
    return memcached_behavior_set(memcached_handle_, flag, data);
  }

  memcached_server_list_st MemcachedServerListAppend(
      memcached_server_list_st ptr, const char* hostname, in_port_t port,
      memcached_return_t* error) override {
    return memcached_server_list_append(ptr, hostname, port, error);
  }

  memcached_return_t MemcachedServerPush(
      const memcached_server_list_st list) override {
    return memcached_server_push(memcached_handle_, list);
  }

  memcached_return_t MemcachedSet(const char* key, size_t key_length,
                                  const char* value, size_t value_length,
                                  time_t expiration, uint32_t flags) override {
    return memcached_set(memcached_handle_, key, key_length, value,
                         value_length, expiration, flags);
  }

  char* MemcachedGet(const char* key, size_t key_length, size_t* value_length,
                     uint32_t* flags, memcached_return_t* error) override {
    return memcached_get(memcached_handle_, key, key_length, value_length,
                         flags, error);
  }

  memcached_return_t MemcachedMget(const char* const* keys,
                                   const size_t* key_length,
                                   size_t number_of_keys) override {
    return memcached_mget(memcached_handle_, keys, key_length, number_of_keys);
  }

  memcached_result_st* MemcachedResultCreate(
      memcached_result_st* result) override {
    return memcached_result_create(memcached_handle_, result);
  }

  memcached_result_st* MemcachedFetchResult(
      memcached_result_st* result, memcached_return_t* error) override {
    return memcached_fetch_result(memcached_handle_, result, error);
  }

  size_t MemcachedResultLength(const memcached_result_st* result) override {
    return memcached_result_length(result);
  }

  const char* MemcachedResultValue(const memcached_result_st* result) override {
    return memcached_result_value(result);
  }

  void MemcachedResultFree(memcached_result_st* ptr) override {
    memcached_result_free(ptr);
  }

  const char* MemcachedStrError(memcached_return_t rc) override {
    return memcached_strerror(memcached_handle_, rc);
  }

  void MemcachedFree() override { memcached_free(memcached_handle_); }

  ~MemcachedDao() override {}

 private:
  memcached_st* memcached_handle_ = nullptr;
};

// \brief A memcached block cache of file contents.
//
// This class should be shared by read-only random access files on a remote
// filesystem (e.g. GCS).
class MemcachedFileBlockCache : public FileBlockCache {
 public:
  MemcachedFileBlockCache(
      const std::vector<MemcachedDaoInterface*>& memcached_daos,
      size_t block_size, size_t max_bytes, uint64 max_staleness,
      const size_t local_cache_size, const std::vector<string>& servers,
      const std::vector<string>& options, BlockFetcher block_fetcher,
      Env* env = Env::Default());

  ~MemcachedFileBlockCache() override;

  // \brief The key type for the file block cache.
  //
  // The file block cache key is a {filename, offset} pair.
  typedef std::pair<string, size_t> Key;

  // Reads `n` bytes from `filename` starting at `offset` into `out`. This
  // method will return:
  //
  // 1) The error from the remote filesystem, if the read from the remote
  //    filesystem failed.
  // 2) PRECONDITION_FAILED if the read from the remote filesystem succeeded,
  //    but the read returned a partial block, and the LRU cache contained a
  //    block at a higher offset (indicating that the partial block should have
  //    been a full block).
  // 3) OUT_OF_RANGE if the read from the remote filesystem succeeded, but
  //    the file contents do not extend past `offset` and thus nothing was
  //    placed in `out`.
  // 4) OK otherwise (i.e. the read succeeded, and at least one byte was placed
  //    in `out`).
  Status Read(const string& filename, size_t offset, size_t n, char* buffer,
              size_t* bytes_transferred) override;

  Status MaybeFetch(int64 client_index, const Key& key, std::vector<char>* data)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Validates the given file signature with the existing file signature in the
  // cache. Returns true if the signature doesn't change or the file doesn't
  // exist before. If the signature changes, update the existing signature with
  // the new one and remove the file from cache.
  bool ValidateAndUpdateFileSignature(const string& filename,
                                      int64 file_signature) override
      ABSL_LOCKS_EXCLUDED(mu_);

  // Removes all cached blocks for `filename`.
  void RemoveFile(const string& filename) override ABSL_LOCKS_EXCLUDED(mu_);

  // Removes all cached data.
  void Flush() override ABSL_LOCKS_EXCLUDED(mu_);

  // Accessors for cache parameters.
  size_t block_size() const override { return block_size_; }
  size_t max_bytes() const override { return max_bytes_; }
  uint64 max_staleness() const override { return 0; }

  // The current size (in bytes) of the cache.
  size_t CacheSize() const override;

  // Returns true if the cache is enabled. If false, the BlockFetcher callback
  // is always executed during Read.
  bool IsCacheEnabled() const override {
    return block_size_ > 0 && max_bytes_ > 0;
  }

  // Method called by the thread in charge of sending memcached set
  // requests. Returns 'true' if it is still active or false if the thread
  // should be deactivated.
  bool ProcessCacheBuffer();

 private:
  // Reader threads place the blocks they want to set for memcached in a queue.
  // There is a thread that consumes that queue and sends memcached set
  // requests.
  int64 AddToCacheBuffer(const string& memc_key, std::vector<char>* data);

  // Configures memcached server list and optional behaviors.
  Status ConfigureMemcachedServers(MemcachedDaoInterface* memcached_dao,
                                   const std::vector<string>& server_names,
                                   const std::vector<string>& options);

  // Constructs a memcached key, a single string from the information in Key.
  string MakeMemcachedKey(const Key& key) ABSL_LOCKS_EXCLUDED(mu_);

  // Creates and returns a memcached data access object.
  bool ConfigureMemcachedDao();

  // The size of the blocks stored in the LRU cache, as well as the size of the
  // reads from the underlying filesystem.
  size_t block_size_;
  // The maximum number of bytes (sum of block sizes) allowed in the LRU cache.
  size_t max_bytes_;
  // Whether to fetch keys with multi-get or not.
  bool use_multi_get_;
  // The callback to read a block from the underlying filesystem.
  const BlockFetcher block_fetcher_;
  // The Env from which we read timestamps.
  Env* const env_;  // not owned

  // Guards access to the block map, LRU list, and cached byte count.
  mutable mutex mu_;

  // A filename->file_signature map.
  std::map<string, int64> file_signature_map_ ABSL_GUARDED_BY(mu_);

  // Configuration data for new memcached handles.
  const std::vector<string> servers_;
  const std::vector<string> options_;

  // Thread-specific key for managing clients.
  std::vector<pthread_key_t> cache_keys_;

  // List of data access object to memcached clients.
  std::vector<MemcachedDaoInterface*> memcached_clients_;

  // Thread in charge of iterating over request queue and sending
  // memcached set requests.
  std::unique_ptr<Thread> thread_;
  // Mutex used for the block memcached set request queue.
  mutable mutex throttler_mu_;
  // Queue of keys to store in memcached.
  std::deque<string> cache_buffer_keys_ ABSL_GUARDED_BY(throttler_mu_);
  // Map of keys in the queue to the block data to store.
  std::map<string, std::unique_ptr<std::vector<char>>> cache_buffer_map_
      ABSL_GUARDED_BY(throttler_mu_);
  // Flags that thread_ should stop sending set requests. This is used during
  // destruction, to avoid destroying the thread and/or its resources while
  // it is still doing work. We can Join the thread after setting this to
  // 'true'.
  bool stop_setter_thread_ ABSL_GUARDED_BY(throttler_mu_) = false;

  // Whether the cache was successfully configured. If this is 'false' then read
  // requests will skip the cache and go to GCS.
  bool configured_ = false;

  mutable mutex get_mu_;
  // The available clients are in a queue. Any new GET request can pop the front
  // of the queue for an available memcached client to use and make its GET
  // request. If the queue is empty then all clients are in use by other reads
  // and thus this request will fallback to fetching the data from GCS without
  // querying the distributed cache.
  std::deque<int64> client_queue_ ABSL_GUARDED_BY(get_mu_);

  // Local cache used to serve small reads. Reads that are smaller than the
  // block size require fetching an entire block from GCS or from the
  // distributed cache. We cache those blocks locally in a small local cache
  // because TF workload characteristics show that small reads are typically
  // received in sequential order and in short order of each other. So a small
  // local cache can prevent too many remote requests.
  std::unique_ptr<MiniBlockCache> local_cache_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_IO_GSTPUFS_MEMCACHED_FILE_BLOCK_CACHE_H_
