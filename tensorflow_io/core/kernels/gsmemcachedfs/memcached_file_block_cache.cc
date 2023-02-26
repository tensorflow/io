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

#include "tensorflow_io/core/kernels/gsmemcachedfs/memcached_file_block_cache.h"

#include <random>

#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/fingerprint.h"

namespace tensorflow {

// Approximate max amount of memory for memcached buffer queue. This queue is
// processed constantly by the "setter" thread so it is typically very small.
// At 128MB block size, 12GB buffer memory size allows for about 100 items,
// though the queue will be almost empty if the setter thread is doing its job.
const int64 kMaxMemcachedSetBufferSize = 13421772800;  // 12 GB

namespace block_cache_util {

double GenerateUniformRandomNumber() {
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int64> uniform_dist(1, 6);
  int64 value = uniform_dist(e1);
  return value * (1.0 / std::numeric_limits<uint64>::max());
}

double GenerateUniformRandomNumberBetween(double a, double b) {
  if (a == b) return a;
  DCHECK_LT(a, b);
  return a + GenerateUniformRandomNumber() * (b - a);
}

int64 ComputeBackoffMicroseconds(int current_retry_attempt, int64 min_delay,
                                 int64 max_delay) {
  DCHECK_GE(current_retry_attempt, 0);

  // This function with the constants below is calculating:
  //
  // (0.4 * min_delay) + (random[0.6,1.0] * min_delay * 1.3^retries)
  //
  // Note that there is an extra truncation that occurs and is documented in
  // comments below.
  constexpr double kBackoffBase = 1.3;
  constexpr double kBackoffRandMult = 0.4;

  // This first term does not vary with current_retry_attempt or a random
  // number. It exists to ensure the final term is >= min_delay
  const double first_term = kBackoffRandMult * min_delay;

  // This is calculating min_delay * 1.3^retries
  double uncapped_second_term = min_delay;
  while (current_retry_attempt > 0 &&
         uncapped_second_term < max_delay - first_term) {
    current_retry_attempt--;
    uncapped_second_term *= kBackoffBase;
  }
  // Note that first_term + uncapped_second_term can exceed max_delay here
  // because of the final multiply by kBackoffBase.  We fix that problem with
  // the min() below.
  double second_term = std::min(uncapped_second_term, max_delay - first_term);

  // This supplies the random jitter to ensure that retried don't cause a
  // thundering herd problem.
  second_term *=
      GenerateUniformRandomNumberBetween(1.0 - kBackoffRandMult, 1.0);

  return std::max(static_cast<int64>(first_term + second_term), min_delay);
}

}  // namespace block_cache_util

namespace {

inline void StreamzRecordCacheHitBlockSize(
    size_t value, tsl::FileBlockCacheStatsInterface* cache_stats) {
  if (cache_stats != nullptr) {
    cache_stats->RecordCacheHitBlockSize(value);
  }
}

inline void StreamzRecordCacheMissBlockSize(
    size_t value, tsl::FileBlockCacheStatsInterface* cache_stats) {
  if (cache_stats != nullptr) {
    cache_stats->RecordCacheMissBlockSize(value);
  }
}

// Manages the collation of cache blocks into a requested buffer.
class BufferCollator {
 public:
  BufferCollator(size_t offset, size_t n, char* buffer, size_t block_size)
      : offset_(offset),
        n_(n),
        buffer_(buffer),
        block_size_(block_size),
        total_bytes_transferred_(0),
        start_(0) {}

  void prepare_collation();

  template <typename T>
  bool splice_buffer(T begin, T end, size_t pos,
                     size_t* total_bytes_transferred) const;

  const std::vector<size_t>& positions() { return positions_; }

 private:
  void add_collation_segment(size_t pos);

  size_t offset_;
  size_t n_;
  char* buffer_;
  size_t block_size_;
  size_t total_bytes_transferred_;
  std::vector<size_t> positions_;
  std::vector<size_t> target_offsets_;
  size_t start_;
};

void BufferCollator::add_collation_segment(size_t pos) {
  positions_.push_back(pos);
  target_offsets_.push_back(total_bytes_transferred_);
  size_t begin = 0;
  if (offset_ > pos) {
    // The block begins before the slice we're reading.
    begin += offset_ - pos;
  }
  auto end = block_size_;
  if (pos + block_size_ > offset_ + n_) {
    // The block extends past the end of the slice we're reading.
    end -= (pos + block_size_) - (offset_ + n_);
  }
  if (begin < end) {
    size_t bytes_to_copy = end - begin;
    total_bytes_transferred_ += bytes_to_copy;
  }
}

void BufferCollator::prepare_collation() {
  // Calculate the block-aligned start and end of the read.
  start_ = block_size_ * (offset_ / block_size_);
  size_t finish = block_size_ * ((offset_ + n_) / block_size_);
  if (finish < offset_ + n_) {
    finish += block_size_;
  }
  total_bytes_transferred_ = 0;
  size_t total_blocks = (finish - start_) / block_size_;
  VLOG(2) << "blocks to fetch: " << total_blocks;
  positions_.clear();
  target_offsets_.clear();
  for (size_t pos = start_; pos < finish; pos += block_size_) {
    add_collation_segment(pos);
  }
}

template <typename T>
bool BufferCollator::splice_buffer(T begin, T end, size_t pos,
                                   size_t* total_bytes_transferred) const {
  auto data_size = end - begin;
  if (offset_ > pos) {
    // The block begins before the slice we're reading.
    begin += offset_ - pos;
  }
  if (pos + data_size > offset_ + n_) {
    // The block extends past the end of the slice we're reading.
    end -= (pos + data_size) - (offset_ + n_);
  }
  if (begin < end) {
    size_t bytes_to_copy = end - begin;
    size_t target_offset = target_offsets_[(pos - start_) / block_size_];
    VLOG(3) << "target_offset == " << target_offset
            << "; bytes_to_copy == " << bytes_to_copy;
    memcpy(&buffer_[target_offset], &*begin, bytes_to_copy);
    *total_bytes_transferred += bytes_to_copy;
  }
  if (data_size < block_size_) {
    // The block was a partial block and thus signals EOF at its upper
    // bound.
    return false;
  }
  return true;
}

Status block_get(MemcachedDaoInterface* memcached_dao, const string& key,
                 std::vector<char>* value,
                 tsl::FileBlockCacheStatsInterface* cache_stats) {
  memcached_return rc;
  size_t value_length;
  char* retrieved_value;
  int attempts = 0;
  auto min_delay = absl::ToInt64Microseconds(absl::Milliseconds(100));
  auto max_delay = absl::ToInt64Microseconds(absl::Seconds(5));
  do {
    uint32_t flags;
    retrieved_value = memcached_dao->MemcachedGet(key.c_str(), key.size(),
                                                  &value_length, &flags, &rc);
    if (rc == MEMCACHED_SUCCESS) {
      value->assign(retrieved_value, retrieved_value + value_length);
      // Export some stats.
      StreamzRecordCacheHitBlockSize(value_length, cache_stats);
      // Any object returned by memcached must be released by the caller.
      free(retrieved_value);
      return OkStatus();
    }

    if (rc == MEMCACHED_TIMEOUT) {
      auto delay = block_cache_util::ComputeBackoffMicroseconds(
          attempts++, min_delay, max_delay);
      if (delay > max_delay) {
        break;
      }
      VLOG(3) << "Timed-out memcache_get sleeping for " << delay;
      absl::SleepFor(absl::Microseconds(delay));
    }
  } while (rc == MEMCACHED_TIMEOUT);
  return errors::NotFound("memcached could not get key: ", key,
                          memcached_dao->MemcachedStrError(rc));
}

Status block_multi_get(MemcachedDaoInterface* memcached_dao,
                       const std::vector<string>& keys) {
  memcached_return rc = MEMCACHED_SUCCESS;
  auto key_ptrs = absl::make_unique<const char*[]>(keys.size());
  auto key_lengths = absl::make_unique<size_t[]>(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    key_ptrs[i] = keys[i].c_str();
    key_lengths[i] = keys[i].size();
  }
  rc = memcached_dao->MemcachedMget(key_ptrs.get(), key_lengths.get(),
                                    keys.size());
  if (rc == MEMCACHED_SUCCESS) {
    return OkStatus();
  }

  return errors::Internal("memcached multi_get failed ",
                          memcached_dao->MemcachedStrError(rc));
}

Status block_set(MemcachedDaoInterface* memcached_dao, const string& key,
                 const std::vector<char>& value) {
  memcached_return rc;
  rc = memcached_dao->MemcachedSet(key.c_str(), key.size(), value.data(),
                                   value.size(), static_cast<time_t>(0),
                                   static_cast<uint32_t>(0));
  if (rc == MEMCACHED_SUCCESS) {
    return OkStatus();
  }

  return errors::Internal("memcached failed to store key ", key,
                          memcached_dao->MemcachedStrError(rc));
}

Status read_with_multi_get(
    const BufferCollator& collator, MemcachedDaoInterface* memcached_dao,
    const std::vector<string>& keys,
    std::map<string, MemcachedFileBlockCache::Key>* claim_checks,
    size_t* total_bytes_transferred,
    tsl::FileBlockCacheStatsInterface* cache_stats) {
  VLOG(2) << "Key multi-get of " << claim_checks->size() << " claims";
  const auto before = absl::Now();
  auto to_claim = claim_checks->size();

  Status mget_status = block_multi_get(memcached_dao, keys);
  TF_RETURN_IF_ERROR(mget_status);

  for (int64 i = 0; i < keys.size(); ++i) {
    memcached_return_t fetch_return;
    memcached_result_st fetch_result;
    memcached_dao->MemcachedResultCreate(&fetch_result);

    auto cleanup = gtl::MakeCleanup(
        [&] { memcached_dao->MemcachedResultFree(&fetch_result); });

    fetch_return = MEMCACHED_SUCCESS;
    memcached_dao->MemcachedFetchResult(&fetch_result, &fetch_return);
    if (fetch_return == MEMCACHED_END) {
      break;
    }

    if (fetch_return != MEMCACHED_SUCCESS) {
      return errors::Internal("memcached fetch failure: ",
                              memcached_dao->MemcachedStrError(fetch_return));
    }

    const size_t data_size =
        memcached_dao->MemcachedResultLength(&fetch_result);
    const char* data_begin = memcached_dao->MemcachedResultValue(&fetch_result);
    if (!data_begin) {
      return errors::Internal("memcached fetch failure: ",
                              memcached_dao->MemcachedStrError(fetch_return));
    }

    const char* key_value = memcached_result_key_value(&fetch_result);
    if (!key_value) {
      return errors::Internal("memcached fetch failure: ",
                              memcached_dao->MemcachedStrError(fetch_return));
    }

    const string claim_key = key_value;
    auto claim = claim_checks->find(claim_key);
    if (claim == claim_checks->end()) {
      return errors::Internal("Could not find claim for ", claim_key);
    }

    const size_t pos = claim->second.second;
    VLOG(2) << "memc fetch of " << claim->first << " -> " << '('
            << claim->second.first << ", " << claim->second.second << ')';
    StreamzRecordCacheHitBlockSize(data_size, cache_stats);

    claim_checks->erase(claim);
    collator.splice_buffer(data_begin, data_begin + data_size, pos,
                           total_bytes_transferred);
  }

  const auto after = absl::Now();
  VLOG(2) << (to_claim - claim_checks->size())
          << " multi-get fetches claimed in " << (after - before);
  return OkStatus();
}

void cache_cleanup(void* tsd) {
  memcached_st* state = reinterpret_cast<memcached_st*>(tsd);
  std::unique_ptr<MemcachedDaoInterface> memcached_dao;
  memcached_dao->MemcachedReset(state);
  memcached_dao->MemcachedFree();
}
}  // anonymous namespace

void RunMemcachedSetter(MemcachedFileBlockCache* cache) {
  while (cache->ProcessCacheBuffer()) {
  }
  LOG(INFO) << "Memcached setter thread is done.";
}

MemcachedFileBlockCache::MemcachedFileBlockCache(
    const std::vector<MemcachedDaoInterface*>& memcached_daos,
    size_t block_size, size_t max_bytes, uint64 max_staleness,
    size_t local_cache_size, const std::vector<string>& servers,
    const std::vector<string>& options, BlockFetcher block_fetcher, Env* env)
    : block_size_(block_size),
      max_bytes_(max_bytes),
      use_multi_get_(false),
      block_fetcher_(std::move(block_fetcher)),
      env_(env),
      servers_(servers),
      options_(options) {
  VLOG(1) << "Entering MemcachedFileBlockCache::MemcachedFileBlockCache";

  if (memcached_daos.size() < 2) {
    LOG(ERROR)
        << "Memcached File Block Cache failed to configure because it was not "
           "given enough clients. It needs at least two, was given "
        << memcached_daos.size();
    return;
  }

  for (int64 i = 0; i < memcached_daos.size(); ++i) {
    pthread_key_t cache_key;
    int stat = pthread_key_create(&cache_key, cache_cleanup);
    if (stat) {
      LOG(ERROR) << "Could not create thread-specific key.  Disabling cache.";
      block_size_ = 0;
      max_bytes_ = 0;
      break;
    }
    cache_keys_.push_back(cache_key);
  }
  VLOG(1) << "Memcached chosen block size is " << block_size_;
  VLOG(1) << "GCS memcached file block cache is "
          << (IsCacheEnabled() ? "enabled" : "disabled");

  {
    mutex_lock lock(get_mu_);
    for (int64 i = 0; i < memcached_daos.size(); ++i) {
      memcached_clients_.emplace_back(memcached_daos[i]);
      if (i > 0) {
        // Add all but the first client, reserved is for setter thread.
        client_queue_.push_back(i);
      }
    }
  }

  configured_ = ConfigureMemcachedDao();
  thread_.reset(env->StartThread(ThreadOptions(), "memcached_memc_setter",
                                 [this] { RunMemcachedSetter(this); }));

  local_cache_ = absl::make_unique<MiniBlockCache>(local_cache_size);
  VLOG(1) << "MemcachedFileBlockCache has a local small reads cache of "
          << local_cache_size << " bytes.";

  VLOG(1) << "Departing MemcachedFileBlockCache::MemcachedFileBlockCache";
}

MemcachedFileBlockCache::~MemcachedFileBlockCache() {
  {
    mutex_lock lock(throttler_mu_);
    stop_setter_thread_ = true;
  }
  thread_.reset();
}

bool MemcachedFileBlockCache::ConfigureMemcachedDao() {
  for (int64 i = 0; i < memcached_clients_.size(); ++i) {
    // First get this threads's handle to a memcached client
    memcached_st* tsd =
        reinterpret_cast<memcached_st*>(pthread_getspecific(cache_keys_[i]));
    if (!tsd) {
      VLOG(1) << "Creating specific memcached handle for " << pthread_self();
      memcached_st* handle = memcached_clients_[i]->MemcachedCreate();
      Status status =
          ConfigureMemcachedServers(memcached_clients_[i], servers_, options_);
      if (!status.ok()) {
        LOG(ERROR) << "Could not configure new memcached handle. status="
                   << status;
        return false;
      }
      tsd = handle;
      if (pthread_setspecific(cache_keys_[i], tsd)) {
        LOG(ERROR) << "Could not set thread-specific data";
        return false;
      }
    }
    // Create and return memcached DAO with the current client handle.
    memcached_clients_[i]->MemcachedReset(tsd);
  }
  return true;
}

Status MemcachedFileBlockCache::ConfigureMemcachedServers(
    MemcachedDaoInterface* memcached_dao,
    const std::vector<string>& server_names,
    const std::vector<string>& options) {
  memcached_server_st* servers = nullptr;
  memcached_return rc;

  // Apply known options if present, and log the unused/unknown ones.
  std::set<string> unused_opts;
  std::copy(options.begin(), options.end(),
            inserter(unused_opts, unused_opts.begin()));

  auto opt = unused_opts.find("MGET");
  if (opt != unused_opts.end()) {
    unused_opts.erase(opt);
    use_multi_get_ = true;
    VLOG(1) << "Turned on use of multi-get (mget)";
  }

  opt = unused_opts.find("NO_BLOCK");
  if (opt != unused_opts.end()) {
    unused_opts.erase(opt);
    rc = memcached_dao->MemcachedBehaviorSet(MEMCACHED_BEHAVIOR_NO_BLOCK, 1);
    if (rc != MEMCACHED_SUCCESS) {
      return errors::Internal("Couldn't configure NO_BLOCK for memcache: ",
                              memcached_dao->MemcachedStrError(rc));
    }
    VLOG(1) << "Turned on NO_BLOCK.";
  }

  opt = unused_opts.find("TCP_NODELAY");
  if (opt != unused_opts.end()) {
    unused_opts.erase(opt);
    rc = memcached_dao->MemcachedBehaviorSet(MEMCACHED_BEHAVIOR_TCP_NODELAY, 1);
    if (rc != MEMCACHED_SUCCESS) {
      return errors::Internal("Couldn't configure TCP_NODELAY for memcache: ",
                              memcached_dao->MemcachedStrError(rc));
    }
    VLOG(1) << "Turned on TCP_NODELAY.";
  }

  opt = unused_opts.find("BINARY_PROTOCOL");
  if (opt != unused_opts.end()) {
    unused_opts.erase(opt);
    rc = memcached_dao->MemcachedBehaviorSet(MEMCACHED_BEHAVIOR_BINARY_PROTOCOL,
                                             1);
    if (rc != MEMCACHED_SUCCESS) {
      return errors::Internal(
          "Couldn't configure BINARY_PROTOCOL for memcache: ",
          memcached_dao->MemcachedStrError(rc));
    }
    VLOG(1) << "Turned on BINARY_PROTOCOL.";
  }

  opt = unused_opts.find("IO_KEY_PREFETCH");
  if (opt != unused_opts.end()) {
    unused_opts.erase(opt);
    rc = memcached_dao->MemcachedBehaviorSet(MEMCACHED_BEHAVIOR_IO_KEY_PREFETCH,
                                             1);
    if (rc != MEMCACHED_SUCCESS) {
      return errors::Internal(
          "Couldn't configure IO_KEY_PREFETCH for memcache: ",
          memcached_dao->MemcachedStrError(rc));
    }
    VLOG(1) << "Turned on IO_KEY_PREFETCH.";
  }

  for (const auto& v : unused_opts) {
    VLOG(1) << "Ignoring unknown option " << v;
  }

  for (const string& name : server_names) {
    servers = memcached_dao->MemcachedServerListAppend(servers, name.c_str(),
                                                       11211, &rc);
    if (rc != MEMCACHED_SUCCESS) {
      return errors::Internal("Couldn't add server name: ", name,
                              memcached_dao->MemcachedStrError(rc));
    }
  }
  rc = memcached_dao->MemcachedServerPush(servers);
  if (rc != MEMCACHED_SUCCESS) {
    return errors::Internal("Couldn't add memcached servers: ",
                            memcached_dao->MemcachedStrError(rc));
  }

  return OkStatus();
}

Status MemcachedFileBlockCache::Read(const string& filename, size_t offset,
                                     size_t n, char* buffer,
                                     size_t* bytes_transferred) {
  *bytes_transferred = 0;
  if (n == 0) {
    return OkStatus();
  }
  VLOG(2) << "original read: offset=" << offset << ", n=" << n
          << ", filename=" << filename;

  if (!IsCacheEnabled() || !configured_) {
    // The cache is effectively disabled, so we pass the read through to the
    // fetcher without breaking it up into blocks.
    return block_fetcher_(filename, offset, n, buffer, bytes_transferred);
  }
  auto start_time = absl::Now();
  BufferCollator collator(offset, n, buffer, block_size_);
  collator.prepare_collation();
  std::vector<string> keys;
  std::map<string, Key> claim_checks;
  for (const auto pos : collator.positions()) {
    const Key key = std::make_pair(filename, pos);
    const string memc_key = MakeMemcachedKey(key);
    keys.push_back(memc_key);
    claim_checks.insert(std::make_pair(memc_key, key));
  }

  string mini_read_key = keys[0];
  bool mini_read = n < block_size_;

  if (mini_read) {
    // Small reads get cached locally since we need to fetch an entire block
    // remotely from either GCS or the distributed cache.
    int64 block_offset = offset - offset % block_size_;
    const Key key = std::make_pair(filename, block_offset);
    mini_read_key = MakeMemcachedKey(key);
    int64 offset_in_block = offset - block_offset;
    if (!local_cache_->Peek(mini_read_key)) {
      local_cache_->Fetching(mini_read_key);
    }
    if (local_cache_->Get(mini_read_key, offset_in_block, n, buffer,
                          bytes_transferred)) {
      return OkStatus();
    }
  }

  size_t total_bytes_transferred = 0;
  bool multi_get = use_multi_get_ && !mini_read;

  if (multi_get) {
    int64 client_index = 0;
    {
      mutex_lock lock(get_mu_);

      if (!client_queue_.empty()) {
        client_index = client_queue_.front();
        client_queue_.pop_front();
      } else {
        LOG(WARNING) << "Memcached client pool is oversaturated. Read will "
                        "skip the block cache.";
      }
    }

    if (client_index > 0) {
      auto before = absl::Now();
      Status mget_status = read_with_multi_get(
          collator, memcached_clients_[client_index], keys, &claim_checks,
          &total_bytes_transferred, cache_stats_);
      auto after = absl::Now();
      VLOG(2) << "memc mget: " << (after - before) << ", status "
              << mget_status;
      mutex_lock lock(get_mu_);
      client_queue_.push_back(client_index);
    }
  }

  // At this point, any claims remaining in claim_checks were not retrievable
  // via multi-get, meaning that they were cache misses.  Each of these should
  // be retried serially, dispatching GCS fetches and setting blocks in the
  // cache.
  VLOG(2) << "Serial fetch of " << claim_checks.size() << " claims";
  // When the cache is completely cold for this region of the GCS file, every
  // one of the requests will be a miss. Filling these requests in random
  // offset order is likely less efficient for GCS than filling them in
  // sequence.  Order the claims by offset.
  std::map<size_t, Key> sorted_claims;
  for (auto ci = claim_checks.begin(); ci != claim_checks.end(); ++ci) {
    sorted_claims.insert(std::make_pair(ci->second.second, ci->second));
  }
  for (auto sc = sorted_claims.begin(); sc != sorted_claims.end(); ++sc) {
    size_t pos = sc->first;
    std::vector<char> data;

    int64 client_index = 0;
    if (!multi_get) {
      mutex_lock lock(get_mu_);
      // Get a client ticket from the pool if available.
      if (!client_queue_.empty()) {
        client_index = client_queue_.front();
        client_queue_.pop_front();
      } else {
        LOG(WARNING) << "Memcached client pool is oversaturated. Read will "
                        "skip the block cache.";
      }
    }

    TF_RETURN_IF_ERROR(MaybeFetch(client_index, sc->second, &data));

    if (client_index > 0) {
      mutex_lock lock(get_mu_);
      // Put client ticket back in the pool.
      client_queue_.push_back(client_index);
    }

    // Copy the relevant portion of the block into the result buffer.
    if (offset >= pos + data.size()) {
      // The requested offset is at or beyond the end of the file. This can
      // happen if `offset` is not block-aligned, and the read returns the
      // last block in the file, which does not extend all the way out to
      // `offset`.
      *bytes_transferred = total_bytes_transferred;
      return errors::OutOfRange("EOF at offset ", offset, " in file ", filename,
                                " at position ", pos, "with data size ",
                                data.size());
    }

    if (mini_read) {
      // Add the fetched block to the local cache when serving small read.
      local_cache_->Add(mini_read_key, data.size(), data.data());
      local_cache_->Fetched(mini_read_key);
    }

    if (!collator.splice_buffer(data.begin(), data.end(), pos,
                                &total_bytes_transferred)) {
      break;
    }
  }
  auto finish_time = absl::Now();
  auto elapsed = finish_time - start_time;
  VLOG(2) << "total_bytes_transferred out " << total_bytes_transferred
          << "; rate "
          << (total_bytes_transferred / absl::ToDoubleSeconds(elapsed))
          << " bytes / second";
  *bytes_transferred = total_bytes_transferred;
  return OkStatus();
}

string MemcachedFileBlockCache::MakeMemcachedKey(const Key& key) {
  // Determine hash key usable by memcached.  This will need to be a
  // string <= 250 characters.  Using a key which is the offset, a slash,
  // and a hash of filename, generation, and block size.  Leaving the offset in
  // the clear assists with debugging.
  int64 file_signature = 0;
  {
    mutex_lock lock(mu_);
    auto it = file_signature_map_.find(key.first);
    if (it != file_signature_map_.end()) {
      file_signature = it->second;
    }
  }
  VLOG(3) << "Key{" << key.first << "," << key.second << "}"
          << " has signature " << file_signature;
  string str_key = strings::StrCat(key.first, file_signature, block_size_);
  Fprint128 print = tensorflow::Fingerprint128(str_key);
  string memc_key = strings::StrCat(key.second, "/", print.high64, print.low64);
  VLOG(3) << "memc_key = " << memc_key;
  return memc_key;
}

Status MemcachedFileBlockCache::MaybeFetch(const int64 client_index,
                                           const Key& key,
                                           std::vector<char>* data) {
  string memc_key = MakeMemcachedKey(key);
  if (client_index > 0 && client_index < memcached_clients_.size()) {
    // Just want to mention that in prod we will likely not be hitting this line
    // since we will most likely be doing Multi-Get instead of Get due to perf.
    auto before = absl::Now();
    Status block_get_status = block_get(memcached_clients_[client_index],
                                        memc_key, data, cache_stats_);
    auto after = absl::Now();
    VLOG(2) << "memc get: " << memc_key << ", " << (after - before)
            << ", status " << block_get_status;
    if (block_get_status.ok()) {
      return block_get_status;
    }
  }

  Status status;
  size_t bytes_transferred;
  char* retrieved_value =
      static_cast<char*>(malloc((block_size_) * sizeof(char)));
  auto before_fetch = absl::Now();
  status = block_fetcher_(key.first, key.second, block_size_, retrieved_value,
                          &bytes_transferred);
  auto after_fetch = absl::Now();
  VLOG(2) << "block_fetcher_: " << (after_fetch - before_fetch) << ", status "
          << status << ", bytes_transferred=" << bytes_transferred;

  data->assign(retrieved_value, retrieved_value + bytes_transferred);

  // Any object returned by memcached must be freed by the caller.
  free(retrieved_value);
  TF_RETURN_IF_ERROR(status);

  if (bytes_transferred > 0) {
    auto before_add = absl::Now();
    int64 size = AddToCacheBuffer(memc_key, data);
    auto after_add = absl::Now();
    VLOG(2) << "Add to memcached queue: memc_key = " << memc_key << ", ("
            << (after_add - before_add) << "), size = " << size;
    // Record some stats.
    StreamzRecordCacheMissBlockSize(bytes_transferred, cache_stats_);
  }

  return status;
}

int64 MemcachedFileBlockCache::AddToCacheBuffer(const string& memc_key,
                                                std::vector<char>* data) {
  mutex_lock lock(throttler_mu_);
  if (cache_buffer_keys_.size() * block_size_ >= kMaxMemcachedSetBufferSize) {
    LOG(WARNING)
        << "MemcachedSet queue is overflowing, cache_buffer_keys_.size = "
        << cache_buffer_keys_.size();
    return cache_buffer_keys_.size();
  }

  if (cache_buffer_map_.find(memc_key) == cache_buffer_map_.end()) {
    cache_buffer_keys_.push_back(memc_key);
    auto page = absl::make_unique<std::vector<char>>();
    page->assign(data->begin(), data->end());
    cache_buffer_map_.emplace(memc_key, page.release());
  }
  return cache_buffer_keys_.size();
}

bool MemcachedFileBlockCache::ProcessCacheBuffer() {
  mutex_lock lock(throttler_mu_);
  if (stop_setter_thread_) {
    return false;
  }
  if (cache_buffer_keys_.empty()) {
    return true;
  }

  const string memc_key = cache_buffer_keys_.front();
  cache_buffer_keys_.pop_front();

  if (cache_buffer_map_.find(memc_key) == cache_buffer_map_.end()) {
    // Log error without failing.
    LOG(ERROR) << "Found inconsistent state in which the block at the front of "
                  "the buffer is not found in the map.";
    return true;
  }

  std::unique_ptr<std::vector<char>> data(
      cache_buffer_map_[memc_key].release());
  throttler_mu_.unlock();

  auto before = absl::Now();
  Status status = block_set(memcached_clients_[0], memc_key, *data);
  auto after = absl::Now();
  VLOG(2) << "memc set: " << memc_key << ", " << (after - before) << ", status "
          << status;

  throttler_mu_.lock();

  if (!status.ok()) {
    cache_buffer_keys_.push_back(memc_key);
    cache_buffer_map_[memc_key] = std::move(data);
  } else {
    data = nullptr;
    cache_buffer_map_.erase(memc_key);
  }
  return true;
}

void MemcachedFileBlockCache::RemoveFile(const string& filename) {
  // No action needed here.  The file-specific key will rotate
  // based on generation number and blocksize, so the memcached servers
  // will simply evict the unused blocks.
}

void MemcachedFileBlockCache::Flush() {
  // If necessary, could send the 'flush_all' command to each server,
  // but allowing normal eviction should be sufficient.
}

size_t MemcachedFileBlockCache::CacheSize() const { return 0; }

bool MemcachedFileBlockCache::ValidateAndUpdateFileSignature(
    const string& filename, int64 file_signature) {
  mutex_lock lock(mu_);
  auto it = file_signature_map_.find(filename);
  if (it != file_signature_map_.end()) {
    if (it->second == file_signature) {
      return true;
    }
    it->second = file_signature;
    return false;
  }
  file_signature_map_[filename] = file_signature;
  return true;
}

}  // namespace tensorflow
