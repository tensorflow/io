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
  daos->Connect(pool, cont,allow_cont_creation, status);
  if(TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }
  rc = daos->Mount();
  if(rc != 0) {
    TF_SetStatus(status, TF_INTERNAL,
                "Error Mounting DFS");
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

void CreateDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  int allow_cont_creation = 1;
  std::string pool,cont,dir_path;
  rc = ParseDFSPath(path, &pool, &cont, &dir_path);
  if(rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }
  daos->Connect(pool, cont,allow_cont_creation, status);
  if(TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }
  rc = daos->Mount();
  if(rc != 0) {
    return;
  }
  dfs_obj_t* parent = NULL;
  dir_path = "/" + dir_path;
  rc = dfs_lookup(daos->daos_fs,dir_path.c_str(),O_RDONLY, &parent, NULL, NULL);
  dfs_release(parent);
  if(!rc) {
    TF_SetStatus(status, TF_ALREADY_EXISTS, "");
    return;
  }

  size_t dir_start = dir_path.rfind("/") + 1;
  std::string dir = dir_path.substr(dir_start);
  std::string parent_path = dir_path.substr(0, dir_start);
  parent = NULL;
  if(parent_path != "/") {
    rc = dfs_lookup(daos->daos_fs,parent_path.c_str(),O_RDONLY, &parent, NULL, NULL);
    if(rc) {
      TF_SetStatus(status, TF_NOT_FOUND, "");
      dfs_release(parent);
      return;
    }
  }

  rc = dfs_mkdir(daos->daos_fs,parent,dir.c_str(),S_IWUSR | S_IRUSR,0);
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL,
                "Error Creating Directory");
  }
  else {
    TF_SetStatus(status, TF_OK, "");
  }

  dfs_release(parent);
}

static void RecursivelyCreateDir(const TF_Filesystem* filesystem,
                                 const char* path, TF_Status* status) {
  int rc;
  std::string pool,cont,dir_path;
  rc = ParseDFSPath(path, &pool, &cont, &dir_path);
  if(rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }
  size_t next_dir = PATH_START;
  std::string dir_string;
  std::string path_string(path);
  do {
    next_dir = path_string.find("/", next_dir);
    dir_string = path_string.substr(0,next_dir);
    if(next_dir != std::string::npos) next_dir++;
    CreateDir(filesystem, dir_string.c_str(), status);
    if((TF_GetCode(status) != TF_OK) && (TF_GetCode(status) != TF_ALREADY_EXISTS)) return;
    TF_SetStatus(status, TF_OK, "");

  } while(next_dir != std::string::npos);



}

void DeleteDir(const TF_Filesystem* filesystem, const char* path,
               bool recursive, TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  int allow_cont_creation = 1;
  std::string pool,cont,dir_path;
  rc = ParseDFSPath(path, &pool, &cont, &dir_path);
  if(rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }
  daos->Connect(pool, cont,allow_cont_creation, status);
  if(TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }
  rc = daos->Mount();
  if(rc != 0) {
    return;
  }

  dfs_obj_t* parent = NULL;
  dir_path = "/" + dir_path;
  rc = dfs_lookup(daos->daos_fs,dir_path.c_str(),O_RDONLY, &parent, NULL, NULL);
  dfs_release(parent);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }


  size_t dir_start = dir_path.rfind("/") + 1;
  std::string dir = dir_path.substr(dir_start);
  std::string parent_path = dir_path.substr(0, dir_start);
  parent = NULL;
  if(parent_path != "/") {
    dfs_lookup(daos->daos_fs,parent_path.c_str(),O_RDONLY, &parent, NULL, NULL);
  }

  rc = dfs_remove(daos->daos_fs,parent,dir.c_str(),recursive,NULL);
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL,
                "Error Deleting Directory");
  }
  else {
    TF_SetStatus(status, TF_OK, "");
  }

}

void DeleteSingleDir(const TF_Filesystem* filesystem, const char* path,
                     TF_Status* status) {
  bool recursive = false;
  DeleteDir(filesystem, path, recursive, status);
}

void RecursivelyDeleteDir(const TF_Filesystem* filesystem, const char* path,
                          uint64_t* undeleted_files,
                          uint64_t* undeleted_dirs, TF_Status* status) {
  bool recursive = true;
  DeleteDir(filesystem, path, recursive, status);
  if(TF_GetCode(status) == TF_NOT_FOUND || TF_GetCode(status) == TF_FAILED_PRECONDITION){
    *undeleted_dirs = 1;
    *undeleted_files = 0;
  }
  else {
    *undeleted_dirs = 0;
    *undeleted_files = 0;
  }
}

bool IsDir(const TF_Filesystem* filesystem, const char* path,
           TF_Status* status) {
  int rc;
  bool is_dir = false;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  int allow_cont_creation = 1;
  std::string pool,cont,file;
  rc = ParseDFSPath(path, &pool, &cont, &file);
  if(rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return is_dir;
  }
  daos->Connect(pool, cont,allow_cont_creation, status);
  if(TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return is_dir;
  }
  rc = daos->Mount();
  if(rc != 0) {
    TF_SetStatus(status, TF_INTERNAL,
                "Error Mounting DFS");
    return is_dir;
  }
  dfs_obj_t* obj = NULL;
  file = "/" + file;
  rc = dfs_lookup(daos->daos_fs,file.c_str(),O_RDONLY, &obj, NULL, NULL);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
  }
  else {
    is_dir = S_ISDIR(obj->mode);
    TF_SetStatus(status, TF_OK, "");
    dfs_release(obj);
  }

  return is_dir;
  
}

int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                    TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  int allow_cont_creation = 1;
  std::string pool,cont,file;
  rc = ParseDFSPath(path, &pool, &cont, &file);
  if(rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return -1;
  }
  daos->Connect(pool, cont,allow_cont_creation, status);
  if(TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return -1;
  }
  rc = daos->Mount();
  if(rc != 0) {
    TF_SetStatus(status, TF_INTERNAL,
                "Error Mounting DFS");
    return -1;
  }
  dfs_obj_t* obj = NULL;
  file = "/" + file;
  rc = dfs_lookup(daos->daos_fs,file.c_str(),O_RDONLY, &obj, NULL, NULL);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return -1;
  }
  else {
    if(S_ISDIR(obj->mode)) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
      return -1;
    }
    TF_SetStatus(status, TF_OK, "");
    daos_size_t size;
    dfs_get_size(daos->daos_fs, obj, &size);
    dfs_release(obj);
    return size;
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
  ops->filesystem_ops->create_dir = tf_dfs_filesystem::CreateDir;
  ops->filesystem_ops->delete_dir = tf_dfs_filesystem::DeleteSingleDir;
  ops->filesystem_ops->recursively_create_dir = tf_dfs_filesystem::RecursivelyCreateDir;
  ops->filesystem_ops->is_directory = tf_dfs_filesystem::IsDir;
  ops->filesystem_ops->delete_recursively = tf_dfs_filesystem::RecursivelyDeleteDir;
  ops->filesystem_ops->get_file_size = tf_dfs_filesystem::GetFileSize;




}

}  // namespace dfs
}  // namespace io
}  // namepsace tensorflow
