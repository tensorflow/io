#include <string>
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_io/core/filesystems/filesystem_plugins.h"
#include "tensorflow_io/core/filesystems/dfs/dfs_filesystem.h"

namespace tensorflow {
namespace io {
namespace dfs {

dfs_t dfsFS;

void ParseDFSPath(const std::string& path, std::string* pool_uuid,
                  std::string* cont_uuid, std::string* filename) {
  //parse DFS path in the format of dfs://<pool_uuid>/<cont_uuid>/<filename>
  size_t pool_start = path.find("://") + 3;
  size_t cont_start = path.find("/", pool_start) + 1;
  size_t file_start = path.find("/", cont_start) + 1;
  *pool_uuid = path.substr(pool_start, cont_start - pool_start - 1);
  *cont_uuid = path.substr(cont_start, file_start - cont_start - 1);
  *filename = path.substr(file_start);
}

int ParseUUID(const std::string& str, uuid_t uuid) {
  return uuid_parse(str.c_str(), uuid);
}

int ConnectPool(uuid_t pool_uuid) {
  return daos_pool_connect(pool_uuid, 0, DAOS_PC_RW, &dfsFS.poh, NULL, NULL);
}

int ConnectContainer(uuid_t cont_uuid, int allow_creation) {
  int rc = daos_cont_open(dfsFS.poh, cont_uuid, DAOS_COO_RW, &dfsFS.coh, NULL, NULL);
  if(rc == -DER_NONEXIST) {
    if(allow_creation) {
      rc = dfs_cont_create(dfsFS.poh, cont_uuid, NULL, &dfsFS.coh, NULL);
    }
  }
  return rc;
}

int DisconnectPool() {
  return daos_pool_disconnect(dfsFS.poh, NULL);
}

int DisconnectContainer() {
  return daos_cont_close(dfsFS.coh, 0);
}

class LibDFS {

};


// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_dfs_filesystem {


typedef struct DAOSFileSystemImplementation {
  LibDFS* libdfs;
  DAOSFileSystemImplementation(TF_Status* status);
  ~DAOSFileSystemImplementation() {
    if (libdfs != nullptr) {
      delete libdfs;
    }
  }
} DAOSFileSystemImplementation;

void Connect(const std::string& path, int allow_cont_creation, TF_Status* status) {
  int rc;
  std::string pool,cont,file;
  ParseDFSPath(path, &pool, &cont, &file);
  uuid_t pool_uuid, cont_uuid;
  ParseUUID(pool, pool_uuid);
  if(pool_uuid == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Error Parsing Pool UUID");
    return;
  }
  ParseUUID(cont, cont_uuid);
  if(cont_uuid == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Error Parsing Container UUID");
    return;
  }

  rc = ConnectPool(pool_uuid);
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Error Connecting to Pool");
    return;
  }

  rc = ConnectContainer(cont_uuid, allow_cont_creation);
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Error Connecting to Container");
    return;
  }

}

void Disconnect(TF_Status* status) {
  int rc;
  rc = DisconnectContainer();
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Error Disconnecting from Container");
    return;
  }

  rc = DisconnectPool();
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL,
                "Error Disconnecting from Pool");
    return;
  }

}



} // namespace tf_dfs_filesystem


void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);




}

}  // namespace dfs
}  // namespace io
}  // namepsace tensorflow
