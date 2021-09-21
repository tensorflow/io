#include "absl/synchronization/mutex.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_io/core/filesystems/filesystem_plugins.h"
#include "tensorflow_io/core/filesystems/dfs/dfs_filesystem.h"

namespace tensorflow {
namespace io {
namespace dfs {


// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_dfs_filesystem {

void Init(TF_Filesystem* filesystem, TF_Status* status) {
  filesystem->plugin_filesystem = new DFS();
  int rc = daos_init();
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL,
                "Error Initializing DAOS API");
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}


void Cleanup(TF_Filesystem* filesystem) {
  daos_fini();
  auto daos =
    static_cast<DFS*>(filesystem->plugin_filesystem);
  delete daos;
}

void PathExists(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  int allow_cont_creation = 1;
  std::string pool,cont,file;
  rc = ParseDFSPath(path, &pool, &cont, &file);
  if(rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }
  daos->Connect(path,allow_cont_creation, status);
  if(TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }
  rc = daos->Mount();
  if(rc != 0) {
    return;
  }
  dfs_obj_t* obj = NULL;
  file = "/" + file;
  rc = dfs_lookup(daos->daos_fs,file.c_str(),O_RDONLY, &obj, NULL, NULL);
  dfs_release(obj);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
  }
  else {
    TF_SetStatus(status, TF_OK, "");
  }
}



} // namespace tf_dfs_filesystem


void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
      plugin_memory_allocate(TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_dfs_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_dfs_filesystem::Cleanup;
  ops->filesystem_ops->path_exists = tf_dfs_filesystem::PathExists;




}

}  // namespace dfs
}  // namespace io
}  // namepsace tensorflow
