
#include "tensorflow_io/core/filesystems/dfs/dfs_utils.h"

std::unordered_map<std::string, daos_size_t> DFS::size_map;

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include <dlfcn.h>
#include <stdio.h>
#undef NDEBUG
#include <cassert>

std::string GetStorageString(uint64_t size) {
  if (size < KILO) {
    return std::to_string(size);
  } else if (size < MEGA) {
    return std::to_string(size / KILO) + "K";
  } else if (size < GEGA) {
    return std::to_string(size / MEGA) + "M";
  } else if (size < TERA) {
    return std::to_string(size / GEGA) + "G";
  } else {
    return std::to_string(size / TERA) + "T";
  }
}

size_t GetStorageSize(std::string size) {
  char size_char = size.back();
  size_t curr_scale = 1;
  switch (size_char) {
    case 'K':
      size.pop_back();
      curr_scale *= 1024;
      return (size_t)atoi(size.c_str()) * curr_scale;
    case 'M':
      size.pop_back();
      curr_scale *= 1024 * 1024;
      return (size_t)atoi(size.c_str()) * curr_scale;
    case 'G':
      size.pop_back();
      curr_scale *= 1024 * 1024 * 1024;
      return (size_t)atoi(size.c_str()) * curr_scale;
    case 'T':
      size.pop_back();
      curr_scale *= 1024 * 1024 * 1024;
      return (size_t)atoi(size.c_str()) * curr_scale * 1024;
    default:
      return atoi(size.c_str());
  }
}

mode_t GetFlags(File_Mode mode) {
  switch (mode) {
    case READ:
      return O_RDONLY;
    case WRITE:
      return O_WRONLY | O_CREAT;
    case APPEND:
      return O_WRONLY | O_APPEND | O_CREAT;
    case READWRITE:
      return O_RDWR | O_CREAT;
    default:
      return -1;
  }
}

int ParseUUID(const std::string& str, uuid_t uuid) {
  return uuid_parse(str.c_str(), uuid);
}

void CopyEntries(char*** entries, std::vector<std::string>& results) {
  *entries = static_cast<char**>(
      tensorflow::io::plugin_memory_allocate(results.size() * sizeof(char*)));

  for (uint32_t i = 0; i < results.size(); i++) {
    (*entries)[i] = static_cast<char*>(tensorflow::io::plugin_memory_allocate(
        results[i].size() * sizeof(char)));
    if (results[i][0] == '/') results[i].erase(0, 1);
    strcpy((*entries)[i], results[i].c_str());
  }
}

bool Match(const std::string& filename, const std::string& pattern) {
  return fnmatch(pattern.c_str(), filename.c_str(), FNM_PATHNAME) == 0;
}

DFS::DFS(TF_Status* status) {
  daos_fs = (dfs_t*)malloc(sizeof(dfs_t));
  daos_fs->mounted = false;

  path_map = {};
  TF_SetStatus(status, TF_OK, "");

  // Try to load the necessary daos libraries.
  libdfs.reset(new libDFS(status));
  if (TF_GetCode(status) != TF_OK) {
    libdfs.reset(nullptr);
    return;
  }

  int rc = libdfs->daos_init();
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error initializing DAOS library");
    return;
  }

  rc = libdfs->daos_eq_create(&mEventQueueHandle);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Error initializing DAOS event queue handle");
    return;
  }
}

DFS::~DFS() {
  free(daos_fs);

  int rc;
  for (auto& kv : path_map) {
    auto* obj = kv.second;
    libdfs->dfs_release(obj);
  }
  rc = libdfs->daos_eq_destroy(mEventQueueHandle, 0);
  assert(rc == 0);
  Unmount();
  ClearConnections();

  libdfs->daos_fini();
  libdfs.reset(nullptr);
}

int DFS::ParseDFSPath(const std::string& path, std::string& pool_string,
                      std::string& cont_string, std::string& filename) {
  size_t pool_start = path.find("://") + 3;
  struct duns_attr_t* attr =
      (struct duns_attr_t*)malloc(sizeof(struct duns_attr_t));
  attr->da_rel_path = NULL;
  attr->da_flags = 1;
  attr->da_no_prefix = true;
  std::string direct_path = "/" + path.substr(pool_start);
  bool has_file_name = true;
  size_t cont_start = path.find("/", pool_start) + 1;
  size_t file_start = path.find("/", cont_start) + 1;
  if (file_start == 0) {
    has_file_name = false;
  }
  std::string pool = path.substr(pool_start, cont_start - pool_start - 1);
  std::string cont = path.substr(cont_start, file_start - cont_start - 1);

  if (pools.find(pool) != pools.end()) {
    pool_string = pool;
    auto* poolInfo = pools[pool];
    if (poolInfo->containers->find(cont) != poolInfo->containers->end()) {
      cont_string = cont;
      if (has_file_name)
        filename = path.substr(file_start);
      else
        filename = "/";
      return 0;
    }
  }
  int rc = libdfs->duns_resolve_path(direct_path.c_str(), attr);
  if (rc == 2) {
    attr->da_rel_path = NULL;
    attr->da_flags = 0;
    attr->da_no_prefix = false;
    direct_path = "daos://" + path.substr(pool_start);
    rc = libdfs->duns_resolve_path(direct_path.c_str(), attr);
    if (rc) return rc;
  }
  pool_string = attr->da_pool;
  cont_string = attr->da_cont;
  filename = attr->da_rel_path == NULL ? "" : attr->da_rel_path;
  libdfs->duns_destroy_attr(attr);
  return 0;
}

int DFS::Setup(const std::string& path, std::string& pool_string,
               std::string& cont_string, std::string& file_path,
               TF_Status* status) {
  int allow_cont_creation = 1;
  int rc;
  rc = ParseDFSPath(path, pool_string, cont_string, file_path);
  if (rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return rc;
  }
  Connect(pool_string, cont_string, allow_cont_creation, status);
  if (TF_GetCode(status) != TF_OK) {
    return -1;
  }
  return Mount();
}

void DFS::Connect(std::string& pool_string, std::string& cont_string,
                  int allow_cont_creation, TF_Status* status) {
  int rc;

  rc = ConnectPool(pool_string, status);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Connecting to Pool");
    return;
  }

  rc = ConnectContainer(cont_string, allow_cont_creation, status);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Connecting to Container");
    return;
  }

  connected = true;

  TF_SetStatus(status, TF_OK, "");
}

void DFS::Disconnect(TF_Status* status) {
  int rc;
  rc = DisconnectContainer(pool.first, container.first);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Disconnecting from Container");
    return;
  }

  rc = DisconnectPool(pool.first);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Disconnecting from Pool");
    return;
  }

  connected = false;

  TF_SetStatus(status, TF_OK, "");
}

int DFS::Mount() {
  int rc = 0;
  if (daos_fs->mounted) {
    if (daos_fs->poh.cookie == pool.second.cookie &&
        daos_fs->coh.cookie == container.second.cookie) {
      return rc;
    }
    rc = Unmount();
    if (rc) return rc;
  }
  return libdfs->dfs_mount(pool.second, container.second, O_RDWR, &daos_fs);
}

int DFS::Unmount() {
  int rc;
  if (!daos_fs->mounted) return 0;
  rc = libdfs->dfs_umount(daos_fs);
  daos_fs = (dfs_t*)malloc(sizeof(dfs_t));
  daos_fs->mounted = false;
  return rc;
}

int DFS::Query() {
  int rc;
  daos_pool_info_t pool_info;
  daos_cont_info_t cont_info;
  if (connected) {
    memset(&pool_info, 'D', sizeof(daos_pool_info_t));
    pool_info.pi_bits = DPI_ALL;
    rc = libdfs->daos_pool_query(pool.second, NULL, &pool_info, NULL, NULL);
    if (rc) return rc;
    rc = libdfs->daos_cont_query(container.second, &cont_info, NULL, NULL);
    if (rc) return rc;
    std::cout << "Pool " << pool.first << " ntarget=" << pool_info.pi_ntargets
              << std::endl;
    std::cout << "Pool space info:" << std::endl;
    std::cout << "- Target(VOS) count:" << pool_info.pi_space.ps_ntargets
              << std::endl;
    std::cout << "- SCM:" << std::endl;
    std::cout << "  Total size: "
              << GetStorageString(pool_info.pi_space.ps_space.s_total[0]);
    std::cout << "  Free: "
              << GetStorageString(pool_info.pi_space.ps_space.s_free[0])
              << std::endl;
    std::cout << "- NVMe:" << std::endl;
    std::cout << "  Total size: "
              << GetStorageString(pool_info.pi_space.ps_space.s_total[1]);
    std::cout << "  Free: "
              << GetStorageString(pool_info.pi_space.ps_space.s_free[1])
              << std::endl;
    std::cout << std::endl
              << "Connected Container: " << container.first << std::endl;

    return 0;
  }

  return -1;
}

int DFS::ClearConnections() {
  int rc;
  rc = Unmount();
  if (rc) return rc;
  for (auto pool_it = pools.cbegin(); pool_it != pools.cend();) {
    for (auto cont_it = (*(*pool_it).second->containers).cbegin();
         cont_it != (*(*pool_it).second->containers).cend();) {
      rc = DisconnectContainer((*pool_it).first, (*cont_it++).first);
      if (rc) return rc;
    }
    rc = DisconnectPool((*pool_it++).first);
    if (rc) return rc;
  }

  return rc;
}

int DFS::dfsDeleteObject(std::string dir_path, bool is_dir, bool recursive,
                         TF_Status* status) {
  dfs_obj_t* temp_obj;
  if (dir_path.back() == '/' && dir_path.size() > 1) dir_path.pop_back();
  if (dir_path.front() != '/') dir_path = "/" + dir_path;
  int rc = dfsPathExists(dir_path, &temp_obj, is_dir);
  if (rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return -1;
  }
  if (!is_dir && S_ISDIR(temp_obj->mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return -1;
  }

  size_t dir_start = dir_path.rfind("/") + 1;
  std::string dir = dir_path.substr(dir_start);
  dfs_obj_t* parent;
  rc = dfsFindParent(dir_path, &parent);
  if (rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return -1;
  }

  rc = libdfs->dfs_remove(daos_fs, parent, dir.c_str(), recursive, NULL);
  if (recursive) {
    path_map.clear();
  } else {
    path_map.erase(dir_path);
  }

  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Deleting Existing Object");
  } else {
    TF_SetStatus(status, TF_OK, "");
  }

  return rc;
}

void DFS::dfsNewFile(std::string file_path, File_Mode file_mode, int flags,
                     dfs_obj_t** obj, TF_Status* status) {
  int rc;
  dfs_obj_t* temp_obj;
  mode_t open_flags;
  if (file_path.back() == '/' && file_path.size() > 1) file_path.pop_back();

  rc = dfsPathExists(file_path, &temp_obj, false);

  if (rc && file_mode == READ) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }

  if (temp_obj != NULL && S_ISDIR(temp_obj->mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }

  if (temp_obj != NULL && file_mode == READ) {
    *obj = temp_obj;
    return;
  }

  if (!rc && file_mode == WRITE) {
    rc = dfsDeleteObject(file_path, false, false, status);
    if (rc) return;
  }

  open_flags = GetFlags(file_mode);

  dfs_obj_t* parent;
  rc = dfsFindParent(file_path, &parent);
  if (rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return;
  }
  if (parent != NULL && !S_ISDIR(parent->mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    return;
  }

  size_t file_start = file_path.rfind("/") + 1;
  std::string file_name = file_path.substr(file_start);
  std::string parent_path = file_path.substr(0, file_start);
  rc = libdfs->dfs_open(daos_fs, parent, file_name.c_str(), flags, open_flags,
                        0, 0, NULL, obj);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Creating Writable File");
    return;
  }
}

int DFS::dfsPathExists(std::string file, dfs_obj_t** obj, bool isDirectory) {
  (*obj) = NULL;
  int rc = 0;
  if (isRoot(file)) {
    return rc;
  }
  if (file.front() != '/') file = "/" + file;
  rc = dfsLookUp(file, obj, isDirectory);
  if (*(obj) == NULL) {
    return ENOENT;
  }

  return rc;
}

int DFS::dfsLookUp(std::string dir_path, dfs_obj_t** obj, bool isDirectory) {
  *obj = NULL;
  dfs_obj_t* parent = NULL;
  int rc;
  if (dir_path.back() == '/' && dir_path.size() > 1) dir_path.pop_back();
  size_t file_start = dir_path.rfind("/") + 1;
  std::string file_path = dir_path.substr(file_start);
  std::string parent_path =
      (file_start > 1) ? dir_path.substr(0, file_start - 1) : "/";
  if (path_map.find(dir_path) != path_map.end()) {
    *obj = path_map[dir_path];
    return 0;
  }
  rc = dfsFindParent(dir_path, &parent);
  if (rc) return rc;
  if (parent != NULL && path_map.count(parent_path) == 0) {
    dfs_obj_t* new_entry = new dfs_obj_t;
    memcpy(new_entry, parent, sizeof(dfs_obj_t));
    path_map[parent_path] = new_entry;
  }

  if (file_path == "/") return rc;
  mode_t open_mode = S_IRUSR | S_IFREG;
  if (isDirectory) {
    open_mode = S_IRUSR | S_IFDIR;
  }
  rc = libdfs->dfs_open(daos_fs, parent, file_path.c_str(), open_mode, O_RDWR,
                        0, 0, 0, obj);
  if (*obj != NULL && path_map.count(dir_path) == 0 && isDirectory) {
    dfs_obj_t* new_entry = new dfs_obj_t;
    memcpy(new_entry, *obj, sizeof(dfs_obj_t));
    path_map[dir_path] = new_entry;
  }
  return rc;
}

int DFS::dfsFindParent(std::string file, dfs_obj_t** parent) {
  (*parent) = NULL;
  if (file.back() == '/' && file.size() > 1) file.pop_back();
  size_t file_start = file.rfind("/") + 1;
  std::string parent_path =
      (file_start > 1) ? file.substr(0, file_start - 1) : "/";
  if (parent_path != "/") {
    int rc = dfsLookUp(parent_path, parent, true);
    if (*(parent) == NULL) return ENOENT;
    return rc;
  } else {
    (*parent) = NULL;
    return 0;
  }
}

int DFS::dfsCreateDir(std::string& dir_path, TF_Status* status) {
  dfs_obj_t* temp_obj;
  int rc;
  rc = dfsPathExists(dir_path, &temp_obj, true);
  if (!rc) {
    TF_SetStatus(status, TF_ALREADY_EXISTS, "");
    return rc;
  }

  size_t dir_start = dir_path.rfind("/") + 1;
  std::string dir = dir_path.substr(dir_start);
  dfs_obj_t* parent;
  rc = dfsFindParent(dir_path, &parent);
  if (rc) {
    TF_SetStatus(status, TF_NOT_FOUND, "");
    return rc;
  }

  rc = libdfs->dfs_mkdir(daos_fs, parent, dir.c_str(), S_IWUSR | S_IRUSR, 0);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Creating Directory");
  } else {
    TF_SetStatus(status, TF_OK, "");
  }

  return rc;
}

bool DFS::isRoot(std::string& file_path) {
  return (file_path.empty() || file_path == "/");
}

int DFS::dfsReadDir(dfs_obj_t* obj, std::vector<std::string>& children) {
  int rc = 0;
  daos_anchor_t anchor = {0};
  uint32_t nr = STACK;
  struct dirent* dirs = (struct dirent*)malloc(nr * sizeof(struct dirent));
  while (!daos_anchor_is_eof(&anchor)) {
    rc = libdfs->dfs_readdir(daos_fs, obj, &anchor, &nr, dirs);
    if (rc) {
      return rc;
    }

    for (uint32_t i = 0; i < nr; i++) {
      children.emplace_back(dirs[i].d_name);
    }
  }

  free(dirs);
  return rc;
}

int DFS::ConnectPool(std::string pool_string, TF_Status* status) {
  int rc = 0;

  if (pools.find(pool_string) != pools.end()) {
    pool.first = pool_string;
    pool.second = pools[pool_string]->poh;
    return rc;
  }

  pool_info_t* po_inf = (pool_info_t*)malloc(sizeof(po_inf));
  rc = libdfs->daos_pool_connect2(pool_string.c_str(), NULL, DAOS_PC_RW,
                                  &(po_inf->poh), NULL, NULL);
  if (rc == 0) {
    pool.first = pool_string;
    pool.second = po_inf->poh;
    po_inf->containers = new std::unordered_map<std::string, daos_handle_t>();
    pools[pool_string] = po_inf;
  }
  return rc;
}

int DFS::ConnectContainer(std::string cont_string, int allow_creation,
                          TF_Status* status) {
  int rc = 0;

  pool_info_t* po_inf = pools[pool.first];
  if (po_inf->containers->find(cont_string) != po_inf->containers->end()) {
    container.first = cont_string;
    container.second = (*po_inf->containers)[cont_string];
    return rc;
  }

  daos_handle_t coh;

  rc = libdfs->daos_cont_open2(pool.second, cont_string.c_str(), DAOS_COO_RW,
                               &coh, NULL, NULL);
  if (rc == -DER_NONEXIST) {
    if (allow_creation) {
      rc = libdfs->dfs_cont_create_with_label(pool.second, cont_string.c_str(),
                                              NULL, NULL, &coh, NULL);
    }
  }
  if (rc == 0) {
    container.first = cont_string;
    container.second = coh;
    (*po_inf->containers)[cont_string] = coh;
  }
  return rc;
}

int DFS::DisconnectPool(std::string pool_string) {
  int rc = 0;
  daos_handle_t poh = pools[pool_string]->poh;
  rc = libdfs->daos_pool_disconnect(poh, NULL);
  if (rc == 0) {
    delete pools[pool_string]->containers;
    free(pools[pool_string]);
    pools.erase(pool_string);
  }
  return rc;
}

int DFS::DisconnectContainer(std::string pool_string, std::string cont_string) {
  int rc = 0;
  daos_handle_t coh = (*pools[pool_string]->containers)[cont_string];
  rc = libdfs->daos_cont_close(coh, nullptr);
  if (rc == 0) {
    pools[pool_string]->containers->erase(cont_string);
  }
  return rc;
}

ReadBuffer::ReadBuffer(size_t aId, DFS *daos, daos_handle_t aEqh, size_t size)
    : id(aId), daos(daos), buffer_size(size), eqh(aEqh) {
  buffer = new char[size];
  buffer_offset = ULONG_MAX;
  event = new daos_event_t;
  int rc = daos->libdfs->daos_event_init(event, eqh, nullptr);
  assert(rc == 0);
}

ReadBuffer::~ReadBuffer() {
  if (event != nullptr) {
    WaitEvent();
    int rc = daos->libdfs->daos_event_fini(event);
    assert(rc == 0);
    delete event;
  }
  if (buffer != nullptr) {
    delete[] buffer;
  }
}

ReadBuffer::ReadBuffer(ReadBuffer&& read_buffer) {
  eqh = read_buffer.eqh;
  buffer_size = read_buffer.buffer_size;
  buffer = std::move(read_buffer.buffer);
  event = std::move(read_buffer.event);
  buffer_offset = ULONG_MAX;
  id = read_buffer.id;
  daos = read_buffer.daos;
  read_buffer.buffer = nullptr;
  read_buffer.event = nullptr;
}

bool ReadBuffer::CacheHit(const size_t pos) {
  return pos >= buffer_offset && (pos < buffer_offset + buffer_size);
}

void ReadBuffer::WaitEvent() {
  bool event_status;
  int rc = daos->libdfs->daos_event_test(event, DAOS_EQ_WAIT, &event_status);
  assert(rc == 0 && event_status == true);
}

int ReadBuffer::ReadAsync(dfs_t* daos_fs, dfs_obj_t* file, const size_t off,
                          const size_t file_size) {
  if (off >= file_size) {
    return 0;
  }
  size_t buffer_actual_size =
      buffer_size > (file_size - off) ? (file_size - off) : buffer_size;
  WaitEvent();
  d_iov_set(&iov, (void*)buffer, buffer_actual_size);
  rsgl.sg_nr = 1;
  rsgl.sg_iovs = &iov;
  buffer_offset = off;
  int rc = daos->libdfs->daos_event_fini(event);
  assert(rc == 0);
  rc = daos->libdfs->daos_event_init(event, eqh, nullptr);
  assert(rc == 0);
  event->ev_error =
      daos->libdfs->dfs_read(daos_fs, file, &rsgl, buffer_offset, &read_size,
                             event);
  return 0;
}

int ReadBuffer::CopyData(char* ret, const size_t ret_offset, const size_t off,
                         const size_t n) {
  WaitEvent();
  if (event->ev_error != DER_SUCCESS) {
    return event->ev_error;
  }
  memcpy(ret + ret_offset, buffer + (off - buffer_offset), n);
  return 0;
}

int64_t ReadBuffer::CopyFromCache(char* ret, const size_t ret_offset,
                                  const size_t off, const size_t n,
                                  const daos_size_t file_size,
                                  TF_Status* status) {
  size_t aRead_size;
  aRead_size = off + n > file_size ? file_size - off : n;
  aRead_size = off + aRead_size > buffer_offset + buffer_size
                   ? buffer_offset + buffer_size - off
                   : aRead_size;
  int rc = CopyData(ret, ret_offset, off, aRead_size);
  if (rc) {
    TF_SetStatusFromIOError(status, rc, "I/O error on dfs_read() call");
    return -1;
  }

  if (off + n > file_size) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "");
  } else {
    TF_SetStatus(status, TF_OK, "");
  }

  return static_cast<int64_t>(aRead_size);
}

static void*
LoadSharedLibrary(const char* library_filename, TF_Status* status)
{
    std::string full_path;
    char* libdir;
    void* handle;

    if ((libdir = std::getenv("TF_IO_DAOS_LIBRARY_DIR")) != nullptr) {
        full_path = libdir;
        if (full_path.back() != '/') full_path.push_back('/');
        full_path.append(library_filename);
        handle = dlopen(full_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            TF_SetStatus(status, TF_OK, "");
            return handle;
        }
    }

    // Check for the library in the installation location used by rpms.
    full_path = "/usr/lib64/";
    full_path += library_filename;
    handle = dlopen(full_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) {
        TF_SetStatus(status, TF_OK, "");
        return handle;
    }

    // Check for the library in the location used when building DAOS fom source.
    full_path = "/opt/daos/lib64/";
    full_path += library_filename;
    handle = dlopen(full_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) {
        TF_SetStatus(status, TF_OK, "");
        return handle;
    }

    std::string error_message =
        absl::StrCat("Library (", library_filename, ") not found: ", dlerror());
    TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
    return nullptr;
}

static void*
GetSymbolFromLibrary(void* handle, const char* symbol_name, TF_Status* status)
{
  if (handle == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "library handle cannot be null");
    return nullptr;
  }
  void* symbol = dlsym(handle, symbol_name);
  if (symbol == nullptr) {
    std::string error_message =
        absl::StrCat("Symbol (", symbol_name, ") not found: ", dlerror());
    TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
    return nullptr;
  }

  TF_SetStatus(status, TF_OK, "");
  return symbol;
}

template <typename R, typename... Args>
void BindFunc(void* handle, const char* name, std::function<R(Args...)>* func,
              TF_Status* status) {
  *func = reinterpret_cast<R (*)(Args...)>(
      GetSymbolFromLibrary(handle, name, status));
}

libDFS::
~libDFS()
{
    if (libdaos_handle_ != nullptr) {
        dlclose(libdaos_handle_);
    }
    if (libdfs_handle_ != nullptr) {
        dlclose(libdfs_handle_);
    }
    if (libduns_handle_ != nullptr) {
        dlclose(libduns_handle_);
    }
}

void libDFS::LoadAndBindDaosLibs(TF_Status* status)
{
#define LOAD_DFS_LIBRARY(handle, library_filename, status)  \
  do {                                                  \
    handle = LoadSharedLibrary(library_filename, status);   \
    if (TF_GetCode(status) != TF_OK) return;            \
  } while (0);

    LOAD_DFS_LIBRARY(libdaos_handle_, "libdaos.so", status);
    LOAD_DFS_LIBRARY(libdfs_handle_, "libdfs.so", status);
    LOAD_DFS_LIBRARY(libduns_handle_, "libduns.so", status);

#undef LOAD_DFS_LIBRARY

#define BIND_DFS_FUNC(handle, function)              \
  do {                                               \
    BindFunc(handle, #function, &function, status);  \
    if (TF_GetCode(status) != TF_OK) return;         \
  } while (0);

    BIND_DFS_FUNC(libdaos_handle_, daos_cont_close);
    BIND_DFS_FUNC(libdaos_handle_, daos_cont_open2);
    BIND_DFS_FUNC(libdaos_handle_, daos_cont_query);
    BIND_DFS_FUNC(libdaos_handle_, daos_event_init);
    BIND_DFS_FUNC(libdaos_handle_, daos_event_fini);
    BIND_DFS_FUNC(libdaos_handle_, daos_event_test);
    BIND_DFS_FUNC(libdaos_handle_, daos_eq_create);
    BIND_DFS_FUNC(libdaos_handle_, daos_eq_destroy);
    BIND_DFS_FUNC(libdaos_handle_, daos_fini);
    BIND_DFS_FUNC(libdaos_handle_, daos_init);
    BIND_DFS_FUNC(libdaos_handle_, daos_pool_connect2);
    BIND_DFS_FUNC(libdaos_handle_, daos_pool_disconnect);
    BIND_DFS_FUNC(libdaos_handle_, daos_pool_query);

    BIND_DFS_FUNC(libdfs_handle_, dfs_cont_create_with_label);
    BIND_DFS_FUNC(libdfs_handle_, dfs_get_size);
    BIND_DFS_FUNC(libdfs_handle_, dfs_mkdir);
    BIND_DFS_FUNC(libdfs_handle_, dfs_mount);
    BIND_DFS_FUNC(libdfs_handle_, dfs_move);
    BIND_DFS_FUNC(libdfs_handle_, dfs_open);
    BIND_DFS_FUNC(libdfs_handle_, dfs_ostat);
    BIND_DFS_FUNC(libdfs_handle_, dfs_read);
    BIND_DFS_FUNC(libdfs_handle_, dfs_readdir);
    BIND_DFS_FUNC(libdfs_handle_, dfs_release);
    BIND_DFS_FUNC(libdfs_handle_, dfs_remove);
    BIND_DFS_FUNC(libdfs_handle_, dfs_umount);
    BIND_DFS_FUNC(libdfs_handle_, dfs_write);

    BIND_DFS_FUNC(libduns_handle_, duns_destroy_attr);
    BIND_DFS_FUNC(libduns_handle_, duns_resolve_path);

#undef BIND_DFS_FUNC
}
