#include "absl/synchronization/mutex.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_io/core/filesystems/filesystem_plugins.h"
#include "tensorflow_io/core/filesystems/dfs/dfs_filesystem.h"

namespace tensorflow {
namespace io {
namespace dfs {


// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {
typedef struct DFSRandomAccessFile {
  std::string dfs_path;
  dfs_t* daos_fs;
  DAOS_FILE daos_file;
  DFSRandomAccessFile(std::string dfs_path, dfs_t* file_system, dfs_obj_t* obj)
      : dfs_path(std::move(dfs_path)) {
    daos_fs = file_system;
    daos_file.file = obj;
  }
} DFSRandomAccessFile;

} // tf_random_access_file



// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {
typedef struct DFSWritableFile {
  std::string dfs_path;
  dfs_t* daos_fs;
  DAOS_FILE daos_file;
  DFSWritableFile(std::string dfs_path, dfs_t* file_system, dfs_obj_t* obj)
      : dfs_path(std::move(dfs_path)) {
    daos_fs = file_system;
    daos_file.file = obj;
  }
} DFSWritableFile;

} //tf_writable_file


// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_dfs_filesystem {

void Init(TF_Filesystem* filesystem, TF_Status* status) {
  filesystem->plugin_filesystem = new DFS();
  auto daos =
    static_cast<DFS*>(filesystem->plugin_filesystem);
  int rc = daos->dfsInit();
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL,
                "Error Initializing DAOS API");
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}


void Cleanup(TF_Filesystem* filesystem) {
  auto daos =
    static_cast<DFS*>(filesystem->plugin_filesystem);
  daos->dfsCleanup();
  delete daos;
}

void NewFile(const TF_Filesystem* filesystem, const char* path,
             mode_t mode, int flags, dfs_obj_t** obj, TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  std::string pool,cont,file_path;
  rc = daos->Setup(path, pool, cont, file_path, status);
  if(rc) return;

  daos->dfsNewFile(file_path, mode, flags, obj, status);
}

void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                     TF_WritableFile* file, TF_Status* status) {

  dfs_obj_t* obj = NULL;
  NewFile(filesystem, path, S_IWUSR | S_IFREG, O_WRONLY | O_CREAT, &obj, status);
  if(TF_GetCode(status) != TF_OK) return;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  file->plugin_file = new tf_writable_file::DFSWritableFile(path,daos->daos_fs,obj);
  TF_SetStatus(status, TF_OK, "");
}

void NewRandomAccessFile(const TF_Filesystem* filesystem, const char* path,
                         TF_RandomAccessFile* file, TF_Status* status) {

  dfs_obj_t* obj = NULL;
  NewFile(filesystem, path, S_IRUSR | S_IFREG, O_RDONLY, &obj, status);
  if(TF_GetCode(status) != TF_OK) return;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  file->plugin_file = new tf_random_access_file::DFSRandomAccessFile(path,daos->daos_fs,obj);
  TF_SetStatus(status, TF_OK, "");
}

void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                       TF_WritableFile* file, TF_Status* status) {

  dfs_obj_t* obj = NULL;
  NewFile(filesystem, path, S_IWUSR | S_IFREG, O_WRONLY | O_CREAT | O_APPEND, &obj, status);
  if(TF_GetCode(status) != TF_OK) return;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  file->plugin_file = new tf_writable_file::DFSWritableFile(path,daos->daos_fs,obj);
  TF_SetStatus(status, TF_OK, "");
}



void PathExists(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  std::string pool,cont,file;
  daos->Setup(path, pool, cont, file, status);
  dfs_obj_t* obj;
  rc = daos->dfsPathExists(file, &obj);
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
  std::string pool,cont,dir_path;
  rc = daos->Setup(path, pool, cont, dir_path, status);
  if(rc) return;

  daos->dfsCreateDir(dir_path, status);
}

static void RecursivelyCreateDir(const TF_Filesystem* filesystem,
                                 const char* path, TF_Status* status) {
  int rc;
  std::string pool,cont,dir_path;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  rc = daos->Setup(path, pool, cont, dir_path, status);
  if(rc) return;


  size_t next_dir = 0;
  std::string dir_string;
  std::string path_string(dir_path);
  do {
    next_dir = path_string.find("/", next_dir);
    dir_string = path_string.substr(0,next_dir);
    if(next_dir != std::string::npos) next_dir++;
    daos->dfsCreateDir(dir_string, status);
    if((TF_GetCode(status) != TF_OK) && (TF_GetCode(status) != TF_ALREADY_EXISTS)) return;
    TF_SetStatus(status, TF_OK, "");

  } while(next_dir != std::string::npos);



}

void DeleteFileSystemEntry(const TF_Filesystem* filesystem, const char* path,
                           bool recursive, bool is_dir, TF_Status* status) {
  int rc;
  std::string pool,cont,dir_path;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  rc = daos->Setup(path, pool, cont, dir_path, status);
  if(rc) return;

  dfs_obj_t* temp_obj;
	rc = daos->dfsPathExists(dir_path, &temp_obj, 0);
	if(rc) {
		TF_SetStatus(status, TF_NOT_FOUND, "");
		return;
	}
  if(!is_dir && S_ISDIR(temp_obj->mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }
  dfs_release(temp_obj);


  size_t dir_start = dir_path.rfind("/") + 1;
	std::string dir = dir_path.substr(dir_start);
	dfs_obj_t* parent;
	rc = daos->dfsFindParent(dir_path, &parent);
	if(rc) {
		TF_SetStatus(status, TF_NOT_FOUND, "");
		return;
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
  bool is_dir = true;
  DeleteFileSystemEntry(filesystem, path, recursive,is_dir, status);
}

void RecursivelyDeleteDir(const TF_Filesystem* filesystem, const char* path,
                          uint64_t* undeleted_files,
                          uint64_t* undeleted_dirs, TF_Status* status) {
  bool recursive = true;
  bool is_dir = true;
  DeleteFileSystemEntry(filesystem, path, recursive, is_dir,status);
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
  std::string pool, cont, file;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  rc = daos->Setup(path, pool, cont, file, status);
  if(rc) return is_dir;


  dfs_obj_t* obj;
  rc = daos->dfsPathExists(file, &obj, 0);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
  }
  else {
    is_dir = S_ISDIR(obj->mode);
    TF_SetStatus(status, TF_OK, "");
  }

  dfs_release(obj);

  return is_dir;
  
}

int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                    TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  std::string pool, cont, file;
  rc = daos->Setup(path, pool, cont, file, status);
  if(rc) return -1;


  dfs_obj_t* obj;
  rc = daos->dfsPathExists(file, &obj, 0);
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

void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  bool recursive = false;
  bool is_dir = false;
  DeleteFileSystemEntry(filesystem, path, recursive, is_dir, status);
}

void RenameFile(const TF_Filesystem* filesystem, const char* src,
                const char* dst, TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  int allow_cont_creation = 1;
  std::string pool_src,cont_src,file_src;
  rc = ParseDFSPath(src, pool_src, cont_src, file_src);
  if(rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }

  std::string pool_dst,cont_dst,file_dst;
  rc = ParseDFSPath(dst, pool_dst, cont_dst, file_dst);
  if(rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }

  if(pool_src != pool_dst || cont_src != cont_dst) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "Non-Matching Pool/Container");
    return;
  }

  daos->Connect(pool_src, cont_src,allow_cont_creation, status);
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

  file_src = "/" + file_src;
  file_dst = "/" + file_dst;

  dfs_obj_t* temp_obj;
  rc = daos->dfsPathExists(file_src, &temp_obj, 0);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }
  else {
    if(S_ISDIR(temp_obj->mode)) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
      dfs_release(temp_obj);
      return;
    }
  }

  dfs_release(temp_obj);
  rc = daos->dfsPathExists(file_dst, &temp_obj, 0);
  if(!rc && S_ISDIR(temp_obj->mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    dfs_release(temp_obj);
    return;
  }

  dfs_release(temp_obj);

  dfs_obj_t* parent_src = NULL;
  size_t src_start = file_src.rfind("/") + 1;
  std::string src_name = file_src.substr(src_start);
  rc = daos->dfsFindParent(file_src, &parent_src);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    dfs_release(parent_src);
    return;
  }

  dfs_obj_t* parent_dst = NULL;
  size_t dst_start = file_dst.rfind("/") + 1;
  std::string dst_name = file_dst.substr(dst_start);
  rc = daos->dfsFindParent(file_dst, &parent_dst);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    dfs_release(parent_dst);
    return;
  }

  if(!S_ISDIR(parent_dst->mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    dfs_release(parent_dst);
    return;
  }

  char* name = (char*)malloc(src_name.size());
  strcpy(name, src_name.c_str());
  char* new_name = (char*)malloc(dst_name.size());
  strcpy(new_name, dst_name.c_str());

  rc = dfs_move(daos->daos_fs, parent_src, name, parent_dst, new_name, NULL);
  free(name);
  free(new_name);
  dfs_release(parent_src);
  dfs_release(parent_dst);
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL, "");
    return;
  }

  TF_SetStatus(status, TF_OK, "");  

}

void Stat(const TF_Filesystem* filesystem, const char* path,
          TF_FileStatistics* stats, TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  std::string pool,cont,dir_path;
  rc = daos->Setup(path, pool, cont, dir_path, status);
  if(rc) return;

  dfs_obj_t* obj;
  rc = daos->dfsPathExists(dir_path, &obj, 0);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }

  if(S_ISDIR(obj->mode)) {
    stats->is_directory = true;
    stats->length = 0;
  }
  else {
    stats->is_directory = false;
    daos_size_t size;
    dfs_get_size(daos->daos_fs, obj, &size);
    stats->length = size;
  }

  struct stat stbuf;

  dfs_ostat(daos->daos_fs, obj, &stbuf);

  stats->mtime_nsec = static_cast<int64_t>(stbuf.st_mtime) * 1e9;

  dfs_release(obj);

  TF_SetStatus(status, TF_OK, "");
  
}

int GetChildren(const TF_Filesystem* filesystem, const char* path,
                char*** entries, TF_Status* status) {
  int rc;
  auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  std::string pool,cont,dir_path;
  rc = daos->Setup(path, pool, cont, dir_path, status);
  if(rc) return -1;

  dfs_obj_t* obj;
  rc = daos->dfsPathExists(dir_path, &obj, 0);
  if(rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return -1;
  }

  if(!S_ISDIR(obj->mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return -1;
  }

  daos_anchor_t anchor = {0};
  uint32_t nr = STACK;
  struct dirent* dirs = (struct dirent*) malloc(nr * sizeof(struct dirent));

  rc = dfs_readdir(daos->daos_fs, obj, &anchor, &nr, dirs);
  if(rc) {
    TF_SetStatus(status, TF_INTERNAL, "");
    return -1;
  }

  *entries = static_cast<char**>(
    plugin_memory_allocate(nr * sizeof((*entries)[0])));

  for(uint32_t i = 0; i < nr; i++) {
    (*entries)[i] = strdup(dirs[i].d_name);
  }

  free(dirs);
  TF_SetStatus(status, TF_OK, "");
  return nr;

}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  return strdup(uri);
}

void FlushCaches(const TF_Filesystem* filesystem) {
   auto daos = 
    static_cast<DFS*>(filesystem->plugin_filesystem);
  daos->ClearConnections();
}
  



} // namespace tf_dfs_filesystem


void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
      plugin_memory_allocate(TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_dfs_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_dfs_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file = tf_dfs_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_dfs_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file = tf_dfs_filesystem::NewAppendableFile;
  ops->filesystem_ops->path_exists = tf_dfs_filesystem::PathExists;
  ops->filesystem_ops->create_dir = tf_dfs_filesystem::CreateDir;
  ops->filesystem_ops->delete_dir = tf_dfs_filesystem::DeleteSingleDir;
  ops->filesystem_ops->recursively_create_dir = tf_dfs_filesystem::RecursivelyCreateDir;
  ops->filesystem_ops->is_directory = tf_dfs_filesystem::IsDir;
  ops->filesystem_ops->delete_recursively = tf_dfs_filesystem::RecursivelyDeleteDir;
  ops->filesystem_ops->get_file_size = tf_dfs_filesystem::GetFileSize;
  ops->filesystem_ops->delete_file = tf_dfs_filesystem::DeleteFile;
  ops->filesystem_ops->rename_file = tf_dfs_filesystem::RenameFile;
  ops->filesystem_ops->stat = tf_dfs_filesystem::Stat;
  ops->filesystem_ops->get_children = tf_dfs_filesystem::GetChildren;
  ops->filesystem_ops->translate_name = tf_dfs_filesystem::TranslateName;
  ops->filesystem_ops->flush_caches = tf_dfs_filesystem::FlushCaches;




}

}  // namespace dfs
}  // namespace io
}  // namepsace tensorflow
