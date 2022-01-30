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

typedef struct DAOS_FILE {
  /** A daos object handle that will contain the file handle of the file
   *  to write and read */
  dfs_obj_t* file;
  /**
   * The offset which represents the number of bytes already written/read
   * using default functions,
   * Modified by seek directly. Incremented naturally by read_dfs_file,
   * write_dfs_file functions.
   * Won't be modified by read_dfs_file_with_offset,
   * write_dfs_file_with_offset.
   * A getter with get_daos_file_offset to obtain the offset.
   */
  long offset;
} DAOS_FILE;

typedef struct pool_info {
  daos_handle_t poh;
  std::map<std::string, daos_handle_t>* containers;
} pool_info_t;

typedef std::pair<std::string, daos_handle_t> id_handle_t;

// enum used while wildcard matching
enum Children_Status { NON_MATCHING, MATCHING_DIR, OK };

std::string FormatStorageSize(uint64_t size);

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

  void dfsNewFile(std::string& file_path, mode_t mode, int flags,
                  dfs_obj_t** obj, TF_Status* status);

  int dfsPathExists(std::string& file, dfs_obj_t** obj, int release_obj = 1);

  int dfsFindParent(std::string& file, dfs_obj_t** parent);

  int dfsCreateDir(std::string& dir_path, TF_Status* status);

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

#endif  // TENSORFLOW_IO_CORE_FILESYSTEMS_DFS_DFS_FILESYSTEM_H_
