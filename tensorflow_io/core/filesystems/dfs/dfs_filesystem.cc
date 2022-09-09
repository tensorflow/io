#include <stdio.h>

#include "tensorflow_io/core/filesystems/dfs/dfs_utils.h"
#undef NDEBUG
#include <cassert>

namespace tensorflow {
namespace io {
namespace dfs {

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {
typedef struct DFSRandomAccessFile {
  dfs_path_t dpath;
  DFS* daos;
  dfs_t* daos_fs;
  dfs_obj_t* daos_file;
  std::vector<ReadBuffer> buffers;
  daos_size_t file_size;
  bool caching;
  size_t buff_size;
  size_t num_of_buffers;
  DFSRandomAccessFile(dfs_path_t* path, dfs_obj_t* obj, daos_handle_t eq_handle)
      : dpath(*path) {
    daos = dpath.getDAOS();
    daos_fs = dpath.getFsys();
    daos_file = obj;

    if (dpath.getCachedSize(file_size) != 0) {
      daos->libdfs->dfs_get_size(daos_fs, obj, &file_size);
      dpath.setCachedSize(file_size);
    }
    if (char* env_caching = std::getenv("TF_IO_DAOS_CACHING")) {
      caching = atoi(env_caching) > 0;
    } else {
      caching = false;
    }

    if (caching) {
      if (char* env_num_of_buffers = std::getenv("TF_IO_DAOS_NUM_OF_BUFFERS")) {
        num_of_buffers = atoi(env_num_of_buffers);
      } else {
        num_of_buffers = NUM_OF_BUFFERS;
      }

      if (char* env_buff_size = std::getenv("TF_IO_DAOS_BUFFER_SIZE")) {
        buff_size = GetStorageSize(env_buff_size);
      } else {
        buff_size = BUFF_SIZE;
      }
      for (size_t i = 0; i < num_of_buffers; i++) {
        buffers.push_back(ReadBuffer(i, daos, eq_handle, buff_size));
      }
    }
  }

  int64_t ReadNoCache(uint64_t offset, size_t n, char* buffer,
                      TF_Status* status) {
    int rc;
    d_sg_list_t rsgl;
    d_iov_t iov;
    d_iov_set(&iov, (void*)buffer, n);
    rsgl.sg_nr = 1;
    rsgl.sg_iovs = &iov;

    daos_size_t read_size;

    rc = daos->libdfs->dfs_read(daos_fs, daos_file, &rsgl, offset, &read_size,
                                NULL);
    if (rc) {
      TF_SetStatus(status, TF_INTERNAL, "");
      return read_size;
    }

    if (read_size != n) {
      TF_SetStatus(status, TF_OUT_OF_RANGE, "");
      return read_size;
    }

    TF_SetStatus(status, TF_OK, "");
    return read_size;
  }
} DFSRandomAccessFile;

void Cleanup(TF_RandomAccessFile* file) {
  int rc = 0;
  auto dfs_file = static_cast<DFSRandomAccessFile*>(file->plugin_file);
  dfs_file->buffers.clear();

  rc = dfs_file->daos->libdfs->dfs_release(dfs_file->daos_file);
  assert(rc == 0);
  dfs_file->daos_fs = nullptr;
  delete dfs_file;
}

int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
             char* ret, TF_Status* status) {
  auto dfs_file = static_cast<DFSRandomAccessFile*>(file->plugin_file);
  if (offset >= dfs_file->file_size) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "");
    return -1;
  }

  if (!dfs_file->caching) {
    return dfs_file->ReadNoCache(offset, n, ret, status);
  }

  size_t ret_offset = 0;
  size_t curr_offset = offset;
  int64_t total_bytes = 0;
  size_t ret_size = offset + n;
  while (curr_offset < ret_size && curr_offset < dfs_file->file_size) {
    int64_t read_bytes = 0;
    for (auto& read_buf : dfs_file->buffers) {
      if (read_buf.CacheHit(curr_offset)) {
        read_bytes = read_buf.CopyFromCache(ret, ret_offset, curr_offset, n,
                                            dfs_file->file_size, status);
        break;
      }
    }

    if (read_bytes < 0) {
      return -1;
    }

    if (read_bytes > 0) {
      curr_offset += read_bytes;
      ret_offset += read_bytes;
      total_bytes += read_bytes;
      n -= read_bytes;
      continue;
    }

    size_t async_offset = curr_offset;
    for (size_t i = 0; i < dfs_file->buffers.size(); i++) {
      if (async_offset > dfs_file->file_size) break;
      dfs_file->buffers[i].ReadAsync(dfs_file->daos_fs, dfs_file->daos_file,
                                     async_offset, dfs_file->file_size);
      async_offset += dfs_file->buff_size;
    }
  }

  return total_bytes;
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {
typedef struct DFSWritableFile {
  dfs_path_t dpath;
  DFS* daos;
  dfs_t* daos_fs;
  dfs_obj_t* daos_file;
  daos_size_t file_size;
  bool size_known;

  DFSWritableFile(dfs_path_t* path, dfs_obj_t* obj) : dpath(*path) {
    daos = dpath.getDAOS();
    daos_fs = dpath.getFsys();
    daos_file = obj;
    size_known = false;
    daos_size_t dummy;  // initialize file_size
    get_file_size(dummy);
  }

  int get_file_size(daos_size_t& size) {
    if (!size_known) {
      int rc = daos->libdfs->dfs_get_size(daos_fs, daos_file, &file_size);
      if (rc != 0) {
        return rc;
      }
      dpath.setCachedSize(file_size);
      size_known = true;
    }
    size = file_size;
    return 0;
  }

  void set_file_size(daos_size_t size) {
    dpath.setCachedSize(size);
    file_size = size;
    size_known = true;
  }

  void unset_file_size(void) {
    dpath.clearCachedSize();
    size_known = false;
  }
} DFSWritableFile;

void Cleanup(TF_WritableFile* file) {
  auto dfs_file = static_cast<DFSWritableFile*>(file->plugin_file);
  dfs_file->daos->libdfs->dfs_release(dfs_file->daos_file);
  dfs_file->daos_fs = nullptr;
  delete dfs_file;
}

void Append(const TF_WritableFile* file, const char* buffer, size_t n,
            TF_Status* status) {
  d_sg_list_t wsgl;
  d_iov_t iov;
  int rc;
  auto dfs_file = static_cast<DFSWritableFile*>(file->plugin_file);

  d_iov_set(&iov, (void*)buffer, n);
  wsgl.sg_nr = 1;
  wsgl.sg_iovs = &iov;

  daos_size_t cur_file_size;
  rc = dfs_file->get_file_size(cur_file_size);
  if (rc != 0) {
    TF_SetStatus(status, TF_INTERNAL, "Cannot determine file size");
    return;
  }

  rc = dfs_file->daos->libdfs->dfs_write(dfs_file->daos_fs, dfs_file->daos_file,
                                         &wsgl, cur_file_size, NULL);
  if (rc) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED, "");
    dfs_file->unset_file_size();
    return;
  }

  dfs_file->set_file_size(cur_file_size + n);
  TF_SetStatus(status, TF_OK, "");
}

int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  auto dfs_file = static_cast<DFSWritableFile*>(file->plugin_file);

  daos_size_t cur_file_size;
  int rc = dfs_file->get_file_size(cur_file_size);
  if (rc != 0) {
    TF_SetStatus(status, TF_INTERNAL, "Cannot determine file size");
    return -1;
  }

  TF_SetStatus(status, TF_OK, "");
  return cur_file_size;
}

void Close(const TF_WritableFile* file, TF_Status* status) {
  auto dfs_file = static_cast<DFSWritableFile*>(file->plugin_file);
  dfs_file->daos->libdfs->dfs_release(dfs_file->daos_file);
  dfs_file->daos_fs = nullptr;
  dfs_file->daos_file = nullptr;
  TF_SetStatus(status, TF_OK, "");
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {
void Cleanup(TF_ReadOnlyMemoryRegion* region) {}

const void* Data(const TF_ReadOnlyMemoryRegion* region) { return nullptr; }

uint64_t Length(const TF_ReadOnlyMemoryRegion* region) { return 0; }

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_dfs_filesystem {

void atexit_handler(void);  // forward declaration

static TF_Filesystem* dfs_filesystem;

void Init(TF_Filesystem* filesystem, TF_Status* status) {
  filesystem->plugin_filesystem = new (std::nothrow) DFS(status);

  // tensorflow never calls Cleanup(), see
  //        https://github.com/tensorflow/tensorflow/issues/27535
  // The workaround is to implement its code via atexit() which in turn
  // requires that a static pointer to the plugin be kept for use at exit time.
  if (TF_GetCode(status) == TF_OK) {
    dfs_filesystem = filesystem;
    std::atexit(atexit_handler);
  }
}

void Cleanup(TF_Filesystem* filesystem) {
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);
  delete daos;
}

void atexit_handler(void) {
  // delete dfs_filesystem;
  Cleanup(dfs_filesystem);
}

void NewFile(const TF_Filesystem* filesystem, const char* path, File_Mode mode,
             int flags, dfs_path_t& dpath, dfs_obj_t** obj, TF_Status* status) {
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  rc = daos->Setup(daos, path, dpath, status);
  if (rc) return;

  daos->dfsNewFile(&dpath, mode, flags, obj, status);
}

void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                     TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");

  dfs_path_t dpath;
  dfs_obj_t* obj = NULL;
  NewFile(filesystem, path, WRITE, S_IRUSR | S_IWUSR | S_IFREG, dpath, &obj,
          status);
  if (TF_GetCode(status) != TF_OK) return;

  file->plugin_file = new tf_writable_file::DFSWritableFile(&dpath, obj);
}

void NewRandomAccessFile(const TF_Filesystem* filesystem, const char* path,
                         TF_RandomAccessFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t dpath;
  dfs_obj_t* obj = NULL;
  NewFile(filesystem, path, READ, S_IRUSR | S_IFREG, dpath, &obj, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto random_access_file = new tf_random_access_file::DFSRandomAccessFile(
      &dpath, obj, daos->mEventQueueHandle);
  if (random_access_file->caching) {
    size_t async_offset = 0;
    for (size_t i = 0; i < random_access_file->num_of_buffers; i++) {
      if (async_offset > random_access_file->file_size) break;
      random_access_file->buffers[i].ReadAsync(
          random_access_file->daos_fs, random_access_file->daos_file,
          async_offset, random_access_file->file_size);
      async_offset += random_access_file->buff_size;
    }
  }
  file->plugin_file = random_access_file;
}

void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                       TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");

  dfs_path_t dpath;
  dfs_obj_t* obj = NULL;
  NewFile(filesystem, path, APPEND, S_IRUSR | S_IWUSR | S_IFREG, dpath, &obj,
          status);
  if (TF_GetCode(status) != TF_OK) return;

  file->plugin_file = new tf_writable_file::DFSWritableFile(&dpath, obj);
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t dpath;
  rc = daos->Setup(daos, path, dpath, status);
  if (rc) return;

  dfs_obj_t* obj;
  rc = daos->dfsLookUp(&dpath, &obj, status);
  if (rc) return;

  rc = daos->libdfs->dfs_release(obj);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "");
  }
}

static void CreateDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t dpath;
  rc = daos->Setup(daos, path, dpath, status);
  if (rc) return;

  daos->dfsCreateDir(&dpath, status);
}

static void RecursivelyCreateDir(const TF_Filesystem* filesystem,
                                 const char* path, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t dpath;
  rc = daos->Setup(daos, path, dpath, status);
  if (rc) return;

  size_t next_dir = 0;
  std::string dir_string;
  std::string path_string = dpath.getRelPath();
  do {
    next_dir = path_string.find("/", next_dir);
    if (next_dir == 0) {
      dpath.setRelPath("/");
    } else {
      dpath.setRelPath(path_string.substr(0, next_dir));
    }
    if (next_dir != std::string::npos) next_dir++;
    TF_SetStatus(status, TF_OK, "");
    daos->dfsCreateDir(&dpath, status);
    if ((TF_GetCode(status) != TF_OK) &&
        (TF_GetCode(status) != TF_ALREADY_EXISTS)) {
      return;
    }
  } while (next_dir != std::string::npos);

  if (TF_GetCode(status) == TF_ALREADY_EXISTS) {
    TF_SetStatus(status, TF_OK, "");  // per modular_filesystem_test suite
  }
}

void DeleteFileSystemEntry(const TF_Filesystem* filesystem, const char* path,
                           bool recursive, bool is_dir, TF_Status* status) {
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t dpath;
  rc = daos->Setup(daos, path, dpath, status);
  if (rc) {
    return;
  }
  daos->dfsDeleteObject(&dpath, is_dir, recursive, status);
}

static void DeleteSingleDir(const TF_Filesystem* filesystem, const char* path,
                            TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  bool recursive = false;
  bool is_dir = true;
  DeleteFileSystemEntry(filesystem, path, recursive, is_dir, status);
}

static void RecursivelyDeleteDir(const TF_Filesystem* filesystem,
                                 const char* path, uint64_t* undeleted_files,
                                 uint64_t* undeleted_dirs, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  bool recursive = true;
  bool is_dir = true;
  DeleteFileSystemEntry(filesystem, path, recursive, is_dir, status);
  if (TF_GetCode(status) == TF_NOT_FOUND ||
      TF_GetCode(status) == TF_FAILED_PRECONDITION) {
    *undeleted_dirs = 1;
    *undeleted_files = 0;
  } else {
    *undeleted_dirs = 0;
    *undeleted_files = 0;
  }
}

// Note: the signature for is_directory() has a bool for the return value, but
// tensorflow does not use this, instead it interprets the status field to get
// the result.  A value of TF_OK indicates that the object is a directory, and
// a value of TF_FAILED_PRECONDITION indicates that the object is a file.  All
// other status values throw an exception.

static bool IsDir(const TF_Filesystem* filesystem, const char* path,
                  TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t dpath;
  rc = daos->Setup(daos, path, dpath, status);
  if (rc) return false;

  dfs_obj_t* obj;
  rc = daos->dfsLookUp(&dpath, &obj, status);
  if (rc) return false;

  bool is_dir = daos->dfsIsDirectory(obj);

  rc = daos->libdfs->dfs_release(obj);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "");
    return false;
  }

  if (!is_dir) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return false;
  }
  return true;
}

static int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                           TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t dpath;
  rc = daos->Setup(daos, path, dpath, status);
  if (rc) return -1;

  daos_size_t size;
  if (dpath.getCachedSize(size) == 0) {
    return size;
  }
  dfs_obj_t* obj;
  rc = daos->dfsLookUp(&dpath, &obj, status);
  if (rc) {
    return -1;
  }

  if (daos->dfsIsDirectory(obj)) {
    daos->libdfs->dfs_release(obj);
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return -1;
  }

  daos->libdfs->dfs_get_size(dpath.getFsys(), obj, &size);
  dpath.setCachedSize(size);

  rc = daos->libdfs->dfs_release(obj);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "");
    return -1;
  }
  return size;
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  bool recursive = false;
  bool is_dir = false;
  DeleteFileSystemEntry(filesystem, path, recursive, is_dir, status);
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t src_dpath;
  rc = daos->Setup(daos, src, src_dpath, status);
  if (rc) return;

  dfs_path_t dst_dpath;
  rc = daos->Setup(daos, dst, dst_dpath, status);
  if (rc) return;

  if (src_dpath.getFsys() != dst_dpath.getFsys()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "Non-Matching Pool/Container");
    return;
  }

  // Source object must exist
  dfs_obj_t* temp_obj;
  rc = daos->dfsLookUp(&src_dpath, &temp_obj, status);
  if (rc) {
    return;
  }

  // Source object cannot be a directory
  bool is_dir = daos->dfsIsDirectory(temp_obj);
  daos_size_t src_size;

  if (is_dir) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
  } else {
    if (src_dpath.getCachedSize(src_size) != 0) {
      daos->libdfs->dfs_get_size(src_dpath.getFsys(), temp_obj, &src_size);
    }
  }

  rc = daos->libdfs->dfs_release(temp_obj);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "");
  }
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  // Destination object may or may not exist, but must not be a directory.
  rc = daos->dfsLookUp(&dst_dpath, &temp_obj, status);
  if (rc) {
    if (TF_GetCode(status) != TF_NOT_FOUND) {
      return;
    }
    TF_SetStatus(status, TF_OK, "");
  } else {
    bool is_dir = daos->dfsIsDirectory(temp_obj);
    rc = daos->libdfs->dfs_release(temp_obj);
    if (rc) {
      TF_SetStatus(status, TF_INTERNAL, "");
      return;
    }
    if (is_dir) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
      return;
    }
  }

  // Open the parent objects.  Note that these are cached directory entries,
  // not to be closed by this function.

  dfs_obj_t* parent_src = NULL;
  rc = daos->dfsFindParent(&src_dpath, &parent_src, status);
  if (rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }

  dfs_obj_t* parent_dst = NULL;
  rc = daos->dfsFindParent(&dst_dpath, &parent_dst, status);
  if (rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }

  if (!daos->dfsIsDirectory(parent_dst)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }

  std::string src_name = src_dpath.getBaseName();
  std::string dst_name = dst_dpath.getBaseName();

  rc = daos->libdfs->dfs_move(src_dpath.getFsys(), parent_src, src_name.c_str(),
                              parent_dst, dst_name.c_str(), NULL);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "");
    return;
  }

  dst_dpath.setCachedSize(src_size);
  src_dpath.clearCachedSize();
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t dpath;
  rc = daos->Setup(daos, path, dpath, status);
  if (rc) return;

  dfs_obj_t* obj;
  rc = daos->dfsLookUp(&dpath, &obj, status);
  if (rc) return;

  struct stat stbuf;
  rc = daos->libdfs->dfs_ostat(dpath.getFsys(), obj, &stbuf);
  if (rc) {
    daos->libdfs->dfs_release(obj);
    TF_SetStatus(status, TF_INTERNAL, "");
    return;
  }

  stats->length = stbuf.st_size;
  stats->mtime_nsec = static_cast<int64_t>(stbuf.st_mtime) * 1e9;
  if (daos->dfsIsDirectory(obj)) {
    stats->is_directory = true;
  } else {
    stats->is_directory = false;
  }

  rc = daos->libdfs->dfs_release(obj);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "");
  }
}

static int GetChildren(const TF_Filesystem* filesystem, const char* path,
                       char*** entries, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  int rc;
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  dfs_path_t dpath;
  rc = daos->Setup(daos, path, dpath, status);
  if (rc) return -1;

  dfs_obj_t* obj;
  rc = daos->dfsLookUp(&dpath, &obj, status);
  if (rc) {
    return -1;
  }

  if (!daos->dfsIsDirectory(obj)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    daos->libdfs->dfs_release(obj);
    return -1;
  }

  std::vector<std::string> children;
  rc = daos->dfsReadDir(dpath.getFsys(), obj, children);
  daos->libdfs->dfs_release(obj);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "");
    return -1;
  }

  uint32_t nr = children.size();

  CopyEntries(entries, children);

  return nr;
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  // Note: this function should be doing the equivalent of the
  // lexically_normalize() function available in newer compilers.
  return strdup(uri);
}

static void FlushCaches(const TF_Filesystem* filesystem) {
  auto daos = static_cast<DFS*>(filesystem->plugin_filesystem);

  daos->clearAllDirCaches();
  daos->clearAllSizeCaches();
}

}  // namespace tf_dfs_filesystem

void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->random_access_file_ops = static_cast<TF_RandomAccessFileOps*>(
      plugin_memory_allocate(TF_RANDOM_ACCESS_FILE_OPS_SIZE));
  ops->random_access_file_ops->cleanup = tf_random_access_file::Cleanup;
  ops->random_access_file_ops->read = tf_random_access_file::Read;

  ops->writable_file_ops = static_cast<TF_WritableFileOps*>(
      plugin_memory_allocate(TF_WRITABLE_FILE_OPS_SIZE));
  ops->writable_file_ops->cleanup = tf_writable_file::Cleanup;
  ops->writable_file_ops->append = tf_writable_file::Append;
  ops->writable_file_ops->tell = tf_writable_file::Tell;
  ops->writable_file_ops->close = tf_writable_file::Close;

  ops->read_only_memory_region_ops = static_cast<TF_ReadOnlyMemoryRegionOps*>(
      plugin_memory_allocate(TF_READ_ONLY_MEMORY_REGION_OPS_SIZE));
  ops->read_only_memory_region_ops->cleanup =
      tf_read_only_memory_region::Cleanup;
  ops->read_only_memory_region_ops->data = tf_read_only_memory_region::Data;
  ops->read_only_memory_region_ops->length = tf_read_only_memory_region::Length;

  ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
      plugin_memory_allocate(TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_dfs_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_dfs_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_dfs_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_dfs_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_dfs_filesystem::NewAppendableFile;
  ops->filesystem_ops->path_exists = tf_dfs_filesystem::PathExists;
  ops->filesystem_ops->create_dir = tf_dfs_filesystem::CreateDir;
  ops->filesystem_ops->delete_dir = tf_dfs_filesystem::DeleteSingleDir;
  ops->filesystem_ops->recursively_create_dir =
      tf_dfs_filesystem::RecursivelyCreateDir;
  ops->filesystem_ops->is_directory = tf_dfs_filesystem::IsDir;
  ops->filesystem_ops->delete_recursively =
      tf_dfs_filesystem::RecursivelyDeleteDir;
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
}  // namespace tensorflow
