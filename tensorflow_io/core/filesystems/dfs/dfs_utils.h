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

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_io/core/filesystems/filesystem_plugins.h"

/** object struct that is instantiated for a DFS open object */
struct dfs_obj {
  /** DAOS object ID */
  daos_obj_id_t oid;
  /** DAOS object open handle */
  daos_handle_t oh;
  /** mode_t containing permissions & type */
  mode_t mode;
  /** open access flags */
  int flags;
  /** DAOS object ID of the parent of the object */
  daos_obj_id_t parent_oid;
  /** entry name of the object in the parent */
  char name[DFS_MAX_NAME + 1];
  union {
    /** Symlink value if object is a symbolic link */
    char* value;
    struct {
      /** Default object class for all entries in dir */
      daos_oclass_id_t oclass;
      /** Default chunk size for all entries in dir */
      daos_size_t chunk_size;
    } d;
  };
};

/** dfs struct that is instantiated for a mounted DFS namespace */
struct dfs {
  /** flag to indicate whether the dfs is mounted */
  bool mounted;
  /** flag to indicate whether dfs is mounted with balanced mode (DTX) */
  bool use_dtx;
  /** lock for threadsafety */
  pthread_mutex_t lock;
  /** uid - inherited from container. */
  uid_t uid;
  /** gid - inherited from container. */
  gid_t gid;
  /** Access mode (RDONLY, RDWR) */
  int amode;
  /** Open pool handle of the DFS */
  daos_handle_t poh;
  /** Open container handle of the DFS */
  daos_handle_t coh;
  /** Object ID reserved for this DFS (see oid_gen below) */
  daos_obj_id_t oid;
  /** superblock object OID */
  daos_obj_id_t super_oid;
  /** Open object handle of SB */
  daos_handle_t super_oh;
  /** Root object info */
  dfs_obj_t root;
  /** DFS container attributes (Default chunk size, oclass, etc.) */
  dfs_attr_t attr;
  /** Optional prefix to account for when resolving an absolute path */
  char* prefix;
  daos_size_t prefix_len;
};

struct dfs_entry {
  /** mode (permissions + entry type) */
  mode_t mode;
  /** Object ID if not a symbolic link */
  daos_obj_id_t oid;
  /* Time of last access */
  time_t atime;
  /* Time of last modification */
  time_t mtime;
  /* Time of last status change */
  time_t ctime;
  /** chunk size of file */
  daos_size_t chunk_size;
  /** Sym Link value */
  char* value;
};

typedef struct pool_info {
  daos_handle_t poh;
  std::map<std::string, daos_handle_t>* containers;
} pool_info_t;

typedef std::pair<std::string, daos_handle_t> id_handle_t;

enum File_Mode { READ, WRITE, APPEND };

std::string GetStorageString(uint64_t size);

size_t GetStorageSize(std::string size);

// parse DFS path in the format of dfs://<pool_uuid>/<cont_uuid>/<filename>
int ParseDFSPath(const std::string& path, std::string& pool_string,
                 std::string& cont_string, std::string& filename);

int ParseUUID(const std::string& str, uuid_t uuid);

class DFS {
 public:
  bool connected;
  dfs_t* daos_fs;
  id_handle_t pool;
  id_handle_t container;
  std::map<std::string, pool_info_t*> pools;

  DFS();

  DFS* Load();

  int dfsInit();

  void dfsCleanup();

  int Setup(const std::string& path, std::string& pool_string,
            std::string& cont_string, std::string& file_path,
            TF_Status* status);

  void Teardown();

  void Connect(std::string& pool_string, std::string& cont_string,
               int allow_cont_creation, TF_Status* status);

  void Disconnect(TF_Status* status);

  int Mount();

  int Unmount();

  int Query();

  int ClearConnections();

  void dfsNewFile(std::string& file_path, File_Mode mode, int flags,
                  dfs_obj_t** obj, TF_Status* status);

  int dfsPathExists(std::string& file, dfs_obj_t** obj, int release_obj = 1);

  int dfsFindParent(std::string& file, dfs_obj_t** parent);

  int dfsCreateDir(std::string& dir_path, TF_Status* status);

  int dfsDeleteObject(std::string& dir_path, bool is_dir, bool recursive,
                      TF_Status* status);

  bool isRoot(std::string& file_path);

  int dfsReadDir(dfs_obj_t* obj, std::vector<std::string>& children);

  ~DFS();

 private:
  bool is_initialized;
  int ConnectPool(std::string pool_string, TF_Status* status);

  int ConnectContainer(std::string cont_string, int allow_creation,
                       TF_Status* status);

  int DisconnectPool(std::string pool_string);

  int DisconnectContainer(std::string pool_string, std::string cont_string);
};

void CopyEntries(char*** entries, std::vector<std::string>& results);

class ReadBuffer {
 public:
  ReadBuffer(size_t id, daos_handle_t eqh, size_t size);

  ReadBuffer(ReadBuffer&&);

  ~ReadBuffer();

  bool CacheHit(const size_t pos);

  void WaitEvent();

  int ReadAsync(dfs_t* dfs, dfs_obj_t* file, const size_t off);

  int CopyData(char* ret, const size_t ret_offset, const size_t offset,
               const size_t n);

  int64_t CopyFromCache(char* ret, const size_t ret_offset, const size_t off,
                    const size_t n, const daos_size_t file_size,
                    TF_Status* status);

 private:
  size_t id;
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
