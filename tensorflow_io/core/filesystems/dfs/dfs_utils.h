#ifndef TENSORFLOW_IO_CORE_FILESYSTEMS_DFS_DFS_FILESYSTEM_H_
#define TENSORFLOW_IO_CORE_FILESYSTEMS_DFS_DFS_FILESYSTEM_H_

#define KILO 1e3
#define MEGA 1e6
#define GEGA 1e9
#define TERA 1e12
#define POOL_START 6
#define CONT_START 43
#define PATH_START 80
#define STACK 24
#define NUM_OF_BUFFERS 256
#define BUFF_SIZE 4 * 1024 * 1024

#include <daos.h>
#include <daos_fs.h>
#include <daos_uns.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <sys/stat.h>

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_io/core/filesystems/filesystem_plugins.h"

typedef std::unordered_map<std::string, dfs_obj_t*> dir_cache_t;
typedef std::unordered_map<std::string, daos_size_t> size_cache_t;

class DFS;

// Class for per-DFS-filesystem state variables, one per container in the
// 'containers' map.
class cont_info_t {
 public:
  daos_handle_t coh;
  DFS* daos;
  std::string pool;
  std::string cont;
  dfs_t* daos_fs;
  dir_cache_t dir_map;
  size_cache_t size_map;
};

typedef struct pool_info {
  daos_handle_t poh;
  std::unordered_map<std::string, cont_info_t*> containers;
} pool_info_t;

// Class for per-DFS-file state variables and common path operations.  State
// includes the filesystem in which the file resides.
class dfs_path_t {
 public:
  dfs_path_t() { cont_info = nullptr; };
  dfs_path_t(cont_info_t* cont_info, std::string rel_path);
  dfs_path_t& operator=(dfs_path_t other);
  DFS* getDAOS(void);
  dfs_t* getFsys(void);
  std::string getFullPath(void);
  std::string getRelPath(void);
  std::string getParentPath(void);
  std::string getBaseName(void);
  void setRelPath(std::string);
  bool isRoot(void);

  dfs_obj_t* getCachedDir(void);
  void setCachedDir(dfs_obj_t* dir_obj);
  void clearCachedDir(void);
  void clearFsysCachedDirs(void);

  int getCachedSize(daos_size_t& size);
  void setCachedSize(daos_size_t size);
  void clearCachedSize(void);
  void clearFsysCachedSizes(void);

 private:
  cont_info_t* cont_info;
  std::string rel_path;
};

typedef std::pair<std::string, daos_handle_t> id_handle_t;

enum File_Mode { READ, WRITE, APPEND, READWRITE };

std::string GetStorageString(uint64_t size);

size_t GetStorageSize(std::string size);

int ParseUUID(const std::string& str, uuid_t uuid);

class libDFS {
 public:
  explicit libDFS(TF_Status* status) { LoadAndBindDaosLibs(status); }

  ~libDFS();

  std::function<int(daos_handle_t, daos_event_t*)> daos_cont_close;

  std::function<int(daos_handle_t, const char*, unsigned int, daos_handle_t*,
                    daos_cont_info_t*, daos_event_t*)>
      daos_cont_open2;

  std::function<int(daos_handle_t, daos_cont_info_t*, daos_prop_t*,
                    daos_event_t*)>
      daos_cont_query;

  std::function<int(daos_event_t*, daos_handle_t, daos_event_t*)>
      daos_event_init;

  std::function<int(daos_event_t*)> daos_event_fini;

  std::function<int(struct daos_event*, int64_t, bool*)> daos_event_test;

  std::function<int(daos_handle_t*)> daos_eq_create;

  std::function<int(daos_handle_t, int)> daos_eq_destroy;

  std::function<int(void)> daos_fini;

  std::function<int(void)> daos_init;

  std::function<int(const char*, const char*, unsigned int, daos_handle_t*,
                    daos_pool_info_t*, daos_event_t*)>
      daos_pool_connect2;

  std::function<int(daos_handle_t, daos_event_t*)> daos_pool_disconnect;

  std::function<int(daos_handle_t, d_rank_list_t*, daos_pool_info_t*,
                    daos_prop_t*, daos_event_t*)>
      daos_pool_query;

  std::function<int(daos_handle_t, const char*, dfs_attr_t*, uuid_t*,
                    daos_handle_t*, dfs_t**)>
      dfs_cont_create_with_label;

  std::function<int(dfs_t*, dfs_obj_t*, int, dfs_obj_t**)> dfs_dup;

  std::function<int(dfs_obj_t*, mode_t*)> dfs_get_mode;

  std::function<int(dfs_t*, dfs_obj_t*, daos_size_t*)> dfs_get_size;

  std::function<int(dfs_t*, const char*, int, dfs_obj_t**, mode_t*,
                    struct stat*)>
      dfs_lookup;

  std::function<int(dfs_t*, dfs_obj_t*, const char*, int, dfs_obj_t**, mode_t*,
                    struct stat*)>
      dfs_lookup_rel;

  std::function<int(dfs_t*, dfs_obj_t*, const char*, mode_t, daos_oclass_id_t)>
      dfs_mkdir;

  std::function<int(daos_handle_t, daos_handle_t, int, dfs_t**)> dfs_mount;

  std::function<int(dfs_t*, dfs_obj_t*, const char*, dfs_obj_t*, const char*,
                    daos_obj_id_t*)>
      dfs_move;

  std::function<int(dfs_t*, dfs_obj_t*, const char*, mode_t, int,
                    daos_oclass_id_t, daos_size_t, const char*, dfs_obj_t**)>
      dfs_open;

  std::function<int(dfs_t*, dfs_obj_t*, struct stat*)> dfs_ostat;

  std::function<int(dfs_t*, dfs_obj_t*, d_sg_list_t*, daos_off_t, daos_size_t*,
                    daos_event_t*)>
      dfs_read;

  std::function<int(dfs_t*, dfs_obj_t*, daos_anchor_t*, uint32_t*,
                    struct dirent*)>
      dfs_readdir;

  std::function<int(dfs_obj_t*)> dfs_release;

  std::function<int(dfs_t*, dfs_obj_t*, const char*, bool, daos_obj_id_t*)>
      dfs_remove;

  std::function<int(dfs_t*)> dfs_umount;

  std::function<int(dfs_t*, dfs_obj_t*, d_sg_list_t*, daos_off_t,
                    daos_event_t*)>
      dfs_write;

  std::function<void(struct duns_attr_t*)> duns_destroy_attr;

  std::function<int(const char*, struct duns_attr_t*)> duns_resolve_path;

 private:
  void LoadAndBindDaosLibs(TF_Status* status);

  void* libdaos_handle_;
  void* libdfs_handle_;
  void* libduns_handle_;
};

// Singlton class for the DFS plugin, containing all its global state.
class DFS {
 public:
  daos_handle_t mEventQueueHandle;
  std::unique_ptr<libDFS> libdfs;
  std::unordered_map<std::string, pool_info_t*> pools;

  explicit DFS(TF_Status* status);

  int ParseDFSPath(const std::string& path, std::string& pool_string,
                   std::string& cont_string, std::string& filename);

  int Setup(DFS* daos, const std::string path, dfs_path_t& dpath,
            TF_Status* status);

  void Connect(DFS* daos, std::string& pool_string, std::string& cont_string,
               int allow_cont_creation, TF_Status* status);

  int Query(id_handle_t pool, id_handle_t container, dfs_t* daos_fs);

  int ClearConnections();

  void clearDirCache(dir_cache_t& dir_cache);

  void clearAllDirCaches(void);

  void clearSizeCache(size_cache_t& size_cache);

  void clearAllSizeCaches(void);

  void dfsNewFile(dfs_path_t* dpath, File_Mode mode, int flags, dfs_obj_t** obj,
                  TF_Status* status);

  int dfsFindParent(dfs_path_t* dpath, dfs_obj_t** obj, TF_Status* status);

  int dfsCreateDir(dfs_path_t* dpath, TF_Status* status);

  int dfsDeleteObject(dfs_path_t* dpath, bool is_dir, bool recursive,
                      TF_Status* status);

  bool dfsIsDirectory(dfs_obj_t* obj);

  int dfsReadDir(dfs_t* daos_fs, dfs_obj_t* obj,
                 std::vector<std::string>& children);

  int dfsLookUp(dfs_path_t* dpath, dfs_obj_t** obj, TF_Status* status);

  dfs_obj_t* lookup_insert_dir(const char* name, mode_t* mode);

  ~DFS();

 private:
  int ConnectPool(std::string pool_string, TF_Status* status);

  int ConnectContainer(DFS* daos, std::string pool_string,
                       std::string cont_string, int allow_creation,
                       TF_Status* status);

  int DisconnectPool(std::string pool_string);

  int DisconnectContainer(std::string pool_string, std::string cont_string);
};

void CopyEntries(char*** entries, std::vector<std::string>& results);

class ReadBuffer {
 public:
  ReadBuffer(size_t id, DFS* daos, daos_handle_t eqh, size_t size);

  ReadBuffer(ReadBuffer&&);

  ~ReadBuffer();

  bool CacheHit(const size_t pos);

  void WaitEvent();

  int ReadAsync(dfs_t* dfs, dfs_obj_t* file, const size_t off,
                const size_t file_size);

  int CopyData(char* ret, const size_t ret_offset, const size_t offset,
               const size_t n);

  int64_t CopyFromCache(char* ret, const size_t ret_offset, const size_t off,
                        const size_t n, const daos_size_t file_size,
                        TF_Status* status);

 private:
  size_t id;
  DFS* daos;
  char* buffer;
  size_t buffer_offset;
  size_t buffer_size;
  daos_handle_t eqh;
  daos_event_t* event;
  d_sg_list_t rsgl;
  d_iov_t iov;
  daos_size_t read_size;
};

#endif  // TENSORFLOW_IO_CORE_FILESYSTEMS_DFS_DFS_FILESYSTEM_H_
