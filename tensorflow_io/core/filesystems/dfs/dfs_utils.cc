#include "tensorflow_io/core/filesystems/dfs/dfs_utils.h"

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
  size_t pool_start = path.find("://") + 3;
  struct duns_attr_t* attr = (struct duns_attr_t*)malloc(sizeof(struct duns_attr_t));
  attr->da_flags = DUNS_NO_PREFIX | DUNS_NO_CHECK_PATH;
  attr->da_no_prefix = true;
  attr->da_rel_path = NULL;
  std::string direct_path = path.substr(pool_start-1);
  int rc = duns_resolve_path(direct_path.c_str(), attr);
  if(rc) {
    return rc;
  }
  pool_string = attr->da_pool;
  cont_string = attr->da_cont;
  filename = attr->da_rel_path == NULL ? "" : attr->da_rel_path;
  return 0;
}

int ParseUUID(const std::string& str, uuid_t uuid) {
  return uuid_parse(str.c_str(), uuid);
}

void CopyEntries(char*** entries, std::vector<std::string>& results) {
	*entries = static_cast<char**>(
			tensorflow::io::plugin_memory_allocate(results.size() * sizeof((*entries)[0])));

	for(uint32_t i = 0; i < results.size(); i++) {
			(*entries)[i] = static_cast<char*>(
			tensorflow::io::plugin_memory_allocate(results[i].size() * sizeof(char)));
			if(results[i][0] == '/') results[i].erase(0,1);
			strcpy((*entries)[i], results[i].c_str());
	}
}

bool Match(const std::string& filename, const std::string& pattern) {
	return fnmatch(pattern.c_str(), filename.c_str(), FNM_PATHNAME) == 0;
}


DFS::DFS() { 
  daos_fs = (dfs_t*)malloc(sizeof(dfs_t));
  daos_fs->mounted = false;
  is_initialized = false;
}

DFS::~DFS() {
  free(daos_fs);
}

DFS* DFS::Load() {
  if(!is_initialized) {
    int rc = dfsInit();
    if(rc) {
      return nullptr;
    }
    is_initialized = true;
  }
  return this;
}

int DFS::dfsInit() {
  return daos_init();
}

void DFS::dfsCleanup() {
  Teardown();
  if(is_initialized){
    daos_fini();
    is_initialized = false;
  }
}

int DFS::Setup(const std::string& path, std::string& pool_string,
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

void DFS::Teardown() {
  Unmount();
  ClearConnections();
}

void DFS::Connect(std::string& pool_string, std::string& cont_string, 
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

void DFS::Disconnect(TF_Status* status) {
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

int DFS::Mount() {
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

int DFS::Unmount() {
int rc;
if(!daos_fs->mounted) return 0;
  rc = dfs_umount(daos_fs);
daos_fs = (dfs_t*)malloc(sizeof(dfs_t));
daos_fs->mounted = false;
  return rc;
}

int DFS::Query() {
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

int DFS::ClearConnections() {
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

void DFS::dfsNewFile(std::string &file_path,mode_t mode, int flags, 
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

int DFS::dfsPathExists(std::string &file, dfs_obj_t **obj, int release_obj) {
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

int DFS::dfsFindParent(std::string &file, dfs_obj_t **parent) {
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

int DFS::dfsCreateDir(std::string &dir_path, TF_Status *status) {
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

bool DFS::isRoot(std::string& file_path) {
  return file_path.empty();
}

int DFS::dfsReadDir(dfs_obj_t* obj, std::vector<std::string>& children) {
  int rc = 0;
  daos_anchor_t anchor = {0};
  uint32_t nr = STACK;
  struct dirent* dirs = (struct dirent*) malloc(nr * sizeof(struct dirent));

  rc = dfs_readdir(daos_fs, obj, &anchor, &nr, dirs);
  if(rc) {
    return rc;
  }

  for(uint32_t i = 0; i < nr; i++) {
    children.push_back(dirs[i].d_name);
  }

  free(dirs);
  return rc;

}


int DFS::ConnectPool(std::string pool_string, TF_Status* status) {
  int rc = 0;

  if(pools.find(pool_string) != pools.end()){
    pool.first = pool_string;
    pool.second = pools[pool_string]->poh;
    return rc;
  }

  pool_info_t* po_inf = (pool_info_t*) malloc(sizeof(po_inf));
  po_inf->containers = new std::map<std::string,daos_handle_t>();
  pools[pool_string] = po_inf;
  rc = daos_pool_connect(pool_string.c_str(), NULL, DAOS_PC_RW, &(po_inf->poh), NULL, NULL);
  if(rc == 0){
    pool.first = pool_string;
    pool.second = po_inf->poh;
  }
  return rc;
}

int DFS::ConnectContainer(std::string cont_string, int allow_creation, TF_Status* status) {
  int rc = 0;

  pool_info_t* po_inf = pools[pool.first];
  if(po_inf->containers->find(cont_string) != po_inf->containers->end()) {
    container.first = cont_string;
    container.second = (*po_inf->containers)[cont_string];
    return rc;
  }

  daos_handle_t coh;

    rc = daos_cont_open(pool.second, cont_string.c_str(), DAOS_COO_RW, &coh, NULL, NULL);
    if(rc == -DER_NONEXIST) {
      if(allow_creation) {
        rc = dfs_cont_create_with_label(pool.second, cont_string.c_str(), NULL, NULL, 
                                        &coh, NULL);
      }
    }
  if(rc == 0){
    container.first = cont_string;
    container.second = coh;
  (*po_inf->containers)[cont_string] = coh;
  }
    return rc;
}

int DFS::DisconnectPool(std:: string pool_string) {
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

int DFS::DisconnectContainer(std::string pool_string, std::string cont_string) {
  int rc = 0;
  daos_handle_t coh = (*pools[pool_string]->containers)[cont_string];
    rc = daos_cont_close(coh, 0);
  if(rc == 0){
    pools[pool_string]->containers->erase(cont_string);
  }
  return rc;
}

void tensorflow::internal::ForEach(int first, int last, const std::function<void(int)>& f) {
		int num_threads = std::min(kNumThreads, last - first);
		thread::ThreadPool threads(Env::Default(), "ForEach", num_threads);
		for (int i = first; i < last; i++) {
			threads.Schedule([f, i] { f(i); });
		}
}