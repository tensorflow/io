#ifndef TENSORFLOW_IO_GSMEMCACHEDFS_GCE_MEMCACHED_SERVER_LIST_PROVIDER_H_  // NOLINT
#define TENSORFLOW_IO_GSMEMCACHEDFS_GCE_MEMCACHED_SERVER_LIST_PROVIDER_H_  // NOLINT

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cloud/compute_engine_metadata_client.h"

namespace tensorflow {

class GceMemcachedServerListProvider {
 public:
  explicit GceMemcachedServerListProvider(
      std::shared_ptr<ComputeEngineMetadataClient> metadata_client);
  virtual ~GceMemcachedServerListProvider();

  Status GetServerList(std::vector<string>* server_list);

  void SetMetadataClient(
      std::shared_ptr<ComputeEngineMetadataClient> metadata_client);

 private:
  std::shared_ptr<ComputeEngineMetadataClient> google_metadata_client_;
  std::vector<string> cached_list_;
  TF_DISALLOW_COPY_AND_ASSIGN(GceMemcachedServerListProvider);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_IO_GSMEMCACHEDFS_GCE_MEMCACHED_SERVER_LIST_PROVIDER_H_
