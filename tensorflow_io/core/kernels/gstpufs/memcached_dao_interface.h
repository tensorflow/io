#ifndef TENSORFLOW_IO_GSTPUFS_MEMCACHED_DAO_INTERFACE_H_
#define TENSORFLOW_IO_GSTPUFS_MEMCACHED_DAO_INTERFACE_H_

#include "libmemcached/memcached.h"

namespace tensorflow {

// Wrapper class which functions as an intermediary between calls to memcached
// and the memcached service. It is useful for testing since we can create
// a mock-memcache object which inherits from this.
class MemcachedDaoInterface {
 public:
  virtual memcached_st* MemcachedCreate() = 0;

  virtual void MemcachedReset(memcached_st* memcached_handle) = 0;

  virtual memcached_return_t MemcachedBehaviorSet(
      const memcached_behavior_t flag, uint64_t data) = 0;

  virtual memcached_server_list_st MemcachedServerListAppend(
      memcached_server_list_st ptr, const char* hostname, in_port_t port,
      memcached_return_t* error) = 0;

  virtual memcached_return_t MemcachedServerPush(
      const memcached_server_list_st list) = 0;

  virtual memcached_return_t MemcachedSet(const char* key, size_t key_length,
                                          const char* value,
                                          size_t value_length,
                                          time_t expiration,
                                          uint32_t flags) = 0;

  virtual char* MemcachedGet(const char* key, size_t key_length,
                             size_t* value_length, uint32_t* flags,
                             memcached_return_t* error) = 0;

  virtual memcached_return_t MemcachedMget(const char* const* keys,
                                           const size_t* key_length,
                                           size_t number_of_keys) = 0;

  virtual memcached_result_st* MemcachedResultCreate(
      memcached_result_st* result) = 0;

  virtual memcached_result_st* MemcachedFetchResult(
      memcached_result_st* result, memcached_return_t* error) = 0;

  virtual size_t MemcachedResultLength(const memcached_result_st* result) = 0;

  virtual const char* MemcachedResultValue(
      const memcached_result_st* result) = 0;

  virtual void MemcachedResultFree(memcached_result_st* ptr) = 0;

  virtual void MemcachedFree() = 0;

  virtual const char* MemcachedStrError(memcached_return_t rc) = 0;

  // Virtual destructor.
  virtual ~MemcachedDaoInterface() = default;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_IO_GSTPUFS_MEMCACHED_DAO_INTERFACE_H_
