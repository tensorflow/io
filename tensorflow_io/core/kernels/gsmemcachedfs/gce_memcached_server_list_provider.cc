#include "tensorflow_io/core/kernels/gsmemcachedfs/gce_memcached_server_list_provider.h"

using namespace tsl;

namespace tensorflow {

namespace {
constexpr char kGceMetadataWorkerNetworkEndpointsPath[] =
    "instance/attributes/worker-network-endpoints";
}  // namespace

GceMemcachedServerListProvider::GceMemcachedServerListProvider(
    std::shared_ptr<ComputeEngineMetadataClient> metadata_client)
    : google_metadata_client_(std::move(metadata_client)) {}

Status GceMemcachedServerListProvider::GetServerList(
    std::vector<string>* server_list) {
  if (!cached_list_.empty()) {
    *server_list = cached_list_;
    return OkStatus();
  }

  std::vector<char> response_buffer;
  TF_RETURN_IF_ERROR(google_metadata_client_->GetMetadata(
      kGceMetadataWorkerNetworkEndpointsPath, &response_buffer));
  StringPiece workers(&response_buffer[0], response_buffer.size());

  std::vector<string> elems = str_util::Split(workers, ",");

  bool success = true;
  for (int64 i = 0; i < elems.size(); i++) {
    std::vector<string> sub_elems = str_util::Split(elems[i], ":");
    if (sub_elems.size() != 3) {
      LOG(ERROR) << "Failed to parse workers server list. Expected 3 items in "
                    "element but found "
                 << sub_elems.size() << " in " << elems[i];
      success = false;
      break;
    }
    string worker_ip = sub_elems[2];
    server_list->push_back(worker_ip);
    cached_list_.push_back(worker_ip);
  }

  if (success) {
    return OkStatus();
  } else {
    return Status(absl::StatusCode::kFailedPrecondition,
                  "Unexpected server list format");
  }
}

void GceMemcachedServerListProvider::SetMetadataClient(
    std::shared_ptr<ComputeEngineMetadataClient> metadata_client) {
  google_metadata_client_.reset(metadata_client.get());
}

GceMemcachedServerListProvider::~GceMemcachedServerListProvider() {}

}  // namespace tensorflow
