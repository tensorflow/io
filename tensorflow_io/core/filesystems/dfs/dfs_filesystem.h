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

#include <sys/stat.h>
#include <fcntl.h>
#include <daos.h>
#include <daos_fs.h>
#include <string>
#include <iostream>
#include <map>

/** object struct that is instantiated for a DFS open object */
struct dfs_obj {
	/** DAOS object ID */
	daos_obj_id_t		oid;
	/** DAOS object open handle */
	daos_handle_t		oh;
	/** mode_t containing permissions & type */
	mode_t			mode;
	/** open access flags */
	int			flags;
	/** DAOS object ID of the parent of the object */
	daos_obj_id_t		parent_oid;
	/** entry name of the object in the parent */
	char			name[DFS_MAX_NAME + 1];
	union {
		/** Symlink value if object is a symbolic link */
		char	*value;
		struct {
			/** Default object class for all entries in dir */
			daos_oclass_id_t        oclass;
			/** Default chunk size for all entries in dir */
			daos_size_t             chunk_size;
		} d;
	};
};

/** dfs struct that is instantiated for a mounted DFS namespace */
struct dfs {
	/** flag to indicate whether the dfs is mounted */
	bool			mounted;
	/** flag to indicate whether dfs is mounted with balanced mode (DTX) */
	bool			use_dtx;
	/** lock for threadsafety */
	pthread_mutex_t		lock;
	/** uid - inherited from container. */
	uid_t			uid;
	/** gid - inherited from container. */
	gid_t			gid;
	/** Access mode (RDONLY, RDWR) */
	int			amode;
	/** Open pool handle of the DFS */
	daos_handle_t		poh;
	/** Open container handle of the DFS */
	daos_handle_t		coh;
	/** Object ID reserved for this DFS (see oid_gen below) */
	daos_obj_id_t		oid;
	/** superblock object OID */
	daos_obj_id_t		super_oid;
	/** Open object handle of SB */
	daos_handle_t		super_oh;
	/** Root object info */
	dfs_obj_t		root;
	/** DFS container attributes (Default chunk size, oclass, etc.) */
	dfs_attr_t		attr;
	/** Optional prefix to account for when resolving an absolute path */
	char			*prefix;
	daos_size_t		prefix_len;
};

struct dfs_entry {
	/** mode (permissions + entry type) */
	mode_t		mode;
	/** Object ID if not a symbolic link */
	daos_obj_id_t	oid;
	/* Time of last access */
	time_t		atime;
	/* Time of last modification */
	time_t		mtime;
	/* Time of last status change */
	time_t		ctime;
	/** chunk size of file */
	daos_size_t	chunk_size;
	/** Sym Link value */
	char		*value;
};

typedef struct DAOS_FILE {
	/** A daos object handle that will contain the file handle of the file
	 *  to write and read */
	dfs_obj_t *file;
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

std::string FormatStorageSize(uint64_t size) {
	if(size < KILO) {
		return std::to_string(size);
	}
	else if(size < MEGA) {
		return std::to_string(size/KILO) + "K";
	}
	else if(size < GEGA) {
		return std::to_string(size/MEGA) + "M";
	}
	else if(size < TERA) {
		return std::to_string(size/GEGA) + "G";
	}
	else {
		return std::to_string(size/TERA) + "T";
	}
}

int ParseDFSPath(const std::string& path, std::string& pool_string,
                  std::string& cont_string, std::string& filename) {
  //parse DFS path in the format of dfs://<pool_uuid>/<cont_uuid>/<filename>
  size_t pool_start = path.find("://") + 3;
  if(pool_start != POOL_START)
	return -1;
  size_t cont_start = path.find("/", pool_start) + 1;
  if(cont_start != CONT_START)
    return -1;
  size_t file_start = path.find("/", cont_start) + 1;
  if(file_start != PATH_START && path.size() > PATH_START)
	return -1;
  pool_string = path.substr(pool_start, cont_start - pool_start - 1);
  cont_string = path.substr(cont_start, file_start - cont_start - 1);
  filename = (file_start == PATH_START)? path.substr(file_start) : "";
  return 0;
}

int ParseUUID(const std::string& str, uuid_t uuid) {
  return uuid_parse(str.c_str(), uuid);
}

class DFS {
  public:
    bool connected;
    dfs_t* daos_fs;
    id_handle_t pool;
    id_handle_t container;
	std::map<std::string,pool_info_t*> pools;

	DFS() { 
	  daos_fs = (dfs_t*)malloc(sizeof(dfs_t));
	  daos_fs->mounted = false;
	}

	int dfsInit() {
	  return daos_init();
	}

	void dfsCleanup() {
	  Teardown();
	  daos_fini();
	}

	int Setup(const std::string& path, std::string& pool_string,
              std::string& cont_string, std::string& file_path, TF_Status* status) {
	  int allow_cont_creation = 1;
	  int rc;
	  rc = ParseDFSPath(path, pool_string, cont_string, file_path);
	  if(rc) {
		TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
		return rc;
	  }
	  Connect(pool_string, cont_string,allow_cont_creation, status);
	  if(TF_GetCode(status) != TF_OK) {
		TF_SetStatus(status, TF_NOT_FOUND, "");
		return rc;
	  }
	  return Mount();
	}

	void Teardown() {
		Unmount();
		ClearConnections();
	}

    void Connect(std::string& pool_string, std::string& cont_string, 
	             int allow_cont_creation, TF_Status* status) {
      int rc;

      rc = ConnectPool(pool_string, status);
      if(rc) {
        TF_SetStatus(status, TF_INTERNAL,
                    "Error Connecting to Pool");
        return;
      }

      rc = ConnectContainer(cont_string, allow_cont_creation, status);
      if(rc) {
        TF_SetStatus(status, TF_INTERNAL,
                    "Error Connecting to Container");
        return;
      }

      connected = true;

	  TF_SetStatus(status, TF_OK, "");

    }

    void Disconnect(TF_Status* status) {
      int rc;
      rc = DisconnectContainer(pool.first, container.first);
      if(rc) {
        TF_SetStatus(status, TF_INTERNAL,
                    "Error Disconnecting from Container");
        return;
      }

      rc = DisconnectPool(pool.first);
      if(rc) {
        TF_SetStatus(status, TF_INTERNAL,
                    "Error Disconnecting from Pool");
        return;
      }

      connected = false;

	  TF_SetStatus(status, TF_OK, "");

    }

    int Mount() {
	  int rc = 0;
	  if(daos_fs->mounted){
		  if(daos_fs->poh.cookie == pool.second.cookie && daos_fs->coh.cookie == container.second.cookie){
			  return rc;
		  }
		  rc = Unmount();
		  if(rc) return rc;
	  }
      return dfs_mount(pool.second, container.second, O_RDWR, &daos_fs);
    }

    int Unmount() {
	  int rc;
	  if(!daos_fs->mounted) return 0;
      rc = dfs_umount(daos_fs);
	  daos_fs = (dfs_t*)malloc(sizeof(dfs_t));
	  daos_fs->mounted = false;
      return rc;
    }

    int Query() {
      int rc;
      daos_pool_info_t pool_info;
      daos_cont_info_t cont_info;
      if(connected) {
        	memset(&pool_info, 'D', sizeof(daos_pool_info_t));
	        pool_info.pi_bits = DPI_ALL;
	        rc = daos_pool_query(pool.second, NULL, &pool_info, NULL, NULL);
          if(rc) return rc;
          rc = daos_cont_query(container.second,&cont_info, NULL, NULL);
          if(rc) return rc;
          std::cout << "Pool " << pool.first << " ntarget=" << pool_info.pi_ntargets << std::endl;
          std::cout << "Pool space info:" << std::endl;
          std::cout << "- Target(VOS) count:" << pool_info.pi_space.ps_ntargets << std::endl;
          std::cout << "- SCM:" << std::endl;
          std::cout << "  Total size: " << FormatStorageSize(pool_info.pi_space.ps_space.s_total[0]);
          std::cout << "  Free: " << FormatStorageSize(pool_info.pi_space.ps_space.s_free[0]) << std::endl;
          std::cout << "- NVMe:" << std::endl;
          std::cout << "  Total size: " << FormatStorageSize(pool_info.pi_space.ps_space.s_total[1]);
          std::cout << "  Free: " << FormatStorageSize(pool_info.pi_space.ps_space.s_free[1]) << std::endl;
          std::cout << std::endl << "Connected Container: " << container.first << std::endl;

          return 0;

      }

      return -1;
    }

	int ClearConnections() {
	  int rc;
	  rc = Unmount();
	  if(rc) return rc;
	  for(auto pool_it = pools.cbegin(); pool_it != pools.cend();) {
	    for(auto cont_it = (*(*pool_it).second->containers).cbegin(); cont_it != (*(*pool_it).second->containers).cend();) {
		  rc = DisconnectContainer((*pool_it).first, (*cont_it++).first);
		  if(rc) return rc;
		  }
		rc = DisconnectPool((*pool_it++).first);
		if(rc) return rc;
	  }

	  return rc;
	}

	void dfsNewFile(std::string &file_path,mode_t mode, int flags, 
	                dfs_obj_t** obj, TF_Status* status) {
		int rc;
		dfs_obj_t* temp_obj;
		rc = dfsPathExists(file_path, &temp_obj, 0);
		if(rc && flags == O_RDONLY) {
			TF_SetStatus(status, TF_NOT_FOUND, "");
			return;
		}

		if(temp_obj != NULL && S_ISDIR(temp_obj->mode)) {
			TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
			dfs_release(temp_obj);
			return;
		}

		if(temp_obj != NULL) {
			dfs_release(temp_obj);
		}

		dfs_obj_t* parent;
		rc =dfsFindParent(file_path, &parent);
		if(rc) {
		TF_SetStatus(status, TF_NOT_FOUND, "");
		dfs_release(parent);
		return;
		}
		if(parent != NULL && !S_ISDIR(parent->mode)) {
		TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
		return;
		}

		size_t file_start = file_path.rfind("/") + 1;
		std::string file_name = file_path.substr(file_start);

		rc = dfs_open(daos_fs, parent, file_name.c_str(), mode, 
						      flags, 0, 0, NULL, obj);
		if(rc) {
			TF_SetStatus(status, TF_INTERNAL, "Error Creating Writable File");
			return;
		}
	}

	int dfsPathExists(std::string &file, dfs_obj_t **obj, int release_obj = 1) {
		(*obj) = NULL;
		int rc = 0;
		if(isRoot(file)) {
			return rc;
		}
  	if(file.front() != '/') file = "/" + file;
  	rc = dfs_lookup(daos_fs,file.c_str(),O_RDONLY, obj, NULL, NULL);
  	if(release_obj) dfs_release(*obj);
		return rc;
	}

	int dfsFindParent(std::string &file, dfs_obj_t **parent) {
		(*parent) = NULL;
		size_t file_start = file.rfind("/") + 1;
		std::string parent_path = file.substr(0, file_start);
		if(parent_path != "/") {
			return dfs_lookup(daos_fs,parent_path.c_str(),O_RDONLY, parent, NULL, NULL);
		}
		else {
			(*parent) = NULL;
			return 0;
		}
	}

	int dfsCreateDir(std::string &dir_path, TF_Status *status) {
		dfs_obj_t* temp_obj;
		int rc;
		rc = dfsPathExists(dir_path, &temp_obj);
		if(!rc) {
			TF_SetStatus(status, TF_ALREADY_EXISTS, "");
			return rc;
		}

		size_t dir_start = dir_path.rfind("/") + 1;
		std::string dir = dir_path.substr(dir_start);
		dfs_obj_t* parent;
		rc = dfsFindParent(dir_path, &parent);
		if(rc) {
			TF_SetStatus(status, TF_NOT_FOUND, "");
			return rc;
		}

		rc = dfs_mkdir(daos_fs,parent,dir.c_str(),S_IWUSR | S_IRUSR,0);
		if(rc) {
			TF_SetStatus(status, TF_INTERNAL,
								  "Error Creating Directory");
		}
		else {
			TF_SetStatus(status, TF_OK, "");
		}

		dfs_release(parent);

		return rc;
	}

	bool isRoot(std::string& file_path) {
		return file_path.empty();
	}

    ~DFS() {
      free(daos_fs);
    }
  private:
    int ConnectPool(std::string pool_string, TF_Status* status) {
	  uuid_t pool_uuid;
	  int rc = 0;
	  rc = ParseUUID(pool_string, pool_uuid);
      if(rc) {
        TF_SetStatus(status, TF_INTERNAL,
                    "Error Parsing Pool UUID");
        return rc;
      }

	  if(pools.find(pool_string) != pools.end()){
		  pool.first = pool_string;
		  pool.second = pools[pool_string]->poh;
		  return rc;
	  }

	  pool_info_t* po_inf = (pool_info_t*) malloc(sizeof(po_inf));
	  po_inf->containers = new std::map<std::string,daos_handle_t>();
	  pools[pool_string] = po_inf;
      rc = daos_pool_connect(pool_uuid, 0, DAOS_PC_RW, &(po_inf->poh), NULL, NULL);
	  if(rc == 0){
	  	pool.first = pool_string;
	  	pool.second = po_inf->poh;
	  }
	  return rc;
    }

    int ConnectContainer(std::string cont_string, int allow_creation, TF_Status* status) {
	  uuid_t cont_uuid;
	  int rc = 0;
	  rc = ParseUUID(cont_string, cont_uuid);
      if(rc) {
        TF_SetStatus(status, TF_INTERNAL,
                    "Error Parsing Container UUID");
        return rc;
      }

	  pool_info_t* po_inf = pools[pool.first];
	  if(po_inf->containers->find(cont_string) != po_inf->containers->end()) {
		  container.first = cont_string;
		  container.second = (*po_inf->containers)[cont_string];
		  return rc;
	  }

	  daos_handle_t coh;

      rc = daos_cont_open(pool.second, cont_uuid, DAOS_COO_RW, &coh, NULL, NULL);
      if(rc == -DER_NONEXIST) {
        if(allow_creation) {
          rc = dfs_cont_create(pool.second, cont_uuid, NULL, &coh, NULL);
        }
      }
	  if(rc == 0){
	  	container.first = cont_string;
	  	container.second = coh;
		(*po_inf->containers)[cont_string] = coh;
	  }
      return rc;
    }

    int DisconnectPool(std:: string pool_string) {
	  int rc = 0;
	  daos_handle_t poh = pools[pool_string]->poh;
      rc = daos_pool_disconnect(poh, NULL);
	  if(rc == 0){
		  delete pools[pool_string]->containers;
		  free(pools[pool_string]);
		  pools.erase(pool_string);
	  }
	  return rc;
    }

    int DisconnectContainer(std::string pool_string, std::string cont_string) {
	   int rc = 0;
	   daos_handle_t coh = (*pools[pool_string]->containers)[cont_string];
       rc = daos_cont_close(coh, 0);
	   if(rc == 0){
		   pools[pool_string]->containers->erase(cont_string);
	   }
	   return rc;
    }


};

#endif  // TENSORFLOW_IO_CORE_FILESYSTEMS_DFS_DFS_FILESYSTEM_H_

