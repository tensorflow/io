
#include "tensorflow_io/core/filesystems/dfs/dfs_utils.h"

#include <dlfcn.h>
#include <stdio.h>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
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
  int rc;
  clearAllDirCaches();
  clearAllSizeCaches();

  rc = libdfs->daos_eq_destroy(mEventQueueHandle, 0);
  assert(rc == 0);
  ClearConnections();

  libdfs->daos_fini();
  libdfs.reset(nullptr);
}

int DFS::ParseDFSPath(const std::string& path, std::string& pool_string,
                      std::string& cont_string, std::string& filename) {
  struct duns_attr_t attr = {0};
  attr.da_flags = DUNS_NO_CHECK_PATH;

  int rc = libdfs->duns_resolve_path(path.c_str(), &attr);
  if (rc == 0) {
    pool_string = attr.da_pool;
    cont_string = attr.da_cont;
    filename = attr.da_rel_path == NULL ? "/" : attr.da_rel_path;
    if (filename.back() == '/' && filename.size() > 1) filename.pop_back();
    libdfs->duns_destroy_attr(&attr);
  }
  return rc;
}

int DFS::Setup(DFS* daos, const std::string path, dfs_path_t& dpath,
               TF_Status* status) {
  int allow_cont_creation = 1;
  int rc;

  std::string pool, cont, rel_path;
  rc = ParseDFSPath(path, pool, cont, rel_path);
  if (rc) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "");
    return rc;
  }
  Connect(daos, pool, cont, allow_cont_creation, status);
  if (TF_GetCode(status) != TF_OK) {
    return -1;
  }

  pool_info_t* po_inf = pools[pool];
  cont_info_t* cont_info = po_inf->containers[cont];
  dfs_path_t res(cont_info, rel_path);
  dpath = res;
  return 0;
}

void DFS::Connect(DFS* daos, std::string& pool_string, std::string& cont_string,
                  int allow_cont_creation, TF_Status* status) {
  int rc;

  rc = ConnectPool(pool_string, status);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Connecting to Pool");
    return;
  }

  rc = ConnectContainer(daos, pool_string, cont_string, allow_cont_creation,
                        status);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Connecting to Container");
    return;
  }

  TF_SetStatus(status, TF_OK, "");
}

int DFS::Query(id_handle_t pool, id_handle_t container, dfs_t* daos_fs) {
  int rc;
  daos_pool_info_t pool_info;
  daos_cont_info_t cont_info;
  if (daos_fs) {
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

  for (;;) {
    auto pool_it = pools.cbegin();
    if (pool_it == pools.cend()) {
      break;
    }
    rc = DisconnectPool((*pool_it++).first);
    if (rc) return rc;
  }
  return 0;
}

void DFS::clearDirCache(dir_cache_t& dir_cache) {
  for (auto kv = dir_cache.begin(); kv != dir_cache.end();) {
    dfs_obj_t* dir = kv->second;
    libdfs->dfs_release(dir);
    kv = dir_cache.erase(kv);
  }
}

void DFS::clearAllDirCaches(void) {
  for (auto pool_it = pools.cbegin(); pool_it != pools.cend();) {
    for (auto cont_it = ((*pool_it).second->containers).cbegin();
         cont_it != ((*pool_it).second->containers).cend();) {
      cont_info_t* cont = (*cont_it).second;
      clearDirCache(cont->dir_map);
      cont_it++;
    }
    pool_it++;
  }
}

void DFS::clearSizeCache(size_cache_t& size_cache) { size_cache.clear(); }

void DFS::clearAllSizeCaches(void) {
  for (auto pool_it = pools.cbegin(); pool_it != pools.cend();) {
    for (auto cont_it = ((*pool_it).second->containers).cbegin();
         cont_it != ((*pool_it).second->containers).cend();) {
      cont_info_t* cont = (*cont_it).second;
      clearSizeCache(cont->size_map);
      cont_it++;
    }
    pool_it++;
  }
}

int DFS::dfsDeleteObject(dfs_path_t* dpath, bool is_dir, bool recursive,
                         TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  dfs_obj_t* temp_obj;

  int rc = dfsLookUp(dpath, &temp_obj, status);
  if (rc) return -1;

  if (dfsIsDirectory(temp_obj)) {
    if (!is_dir) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "Object is a directory");
      libdfs->dfs_release(temp_obj);
      return -1;
    }
  } else {
    if (is_dir && !recursive) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "Object is not a directory");
      libdfs->dfs_release(temp_obj);
      return -1;
    }
  }

  dfs_obj_t* parent;
  rc = dfsFindParent(dpath, &parent, status);
  if (rc) {
    libdfs->dfs_release(temp_obj);
    return -1;
  }

  rc = libdfs->dfs_remove(dpath->getFsys(), parent,
                          dpath->getBaseName().c_str(), recursive, NULL);
  libdfs->dfs_release(temp_obj);
  if (rc) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Error Deleting Existing Object");
    return -1;
  }

  if (is_dir) {
    if (recursive) {
      dpath->clearFsysCachedDirs();
      dpath->clearFsysCachedSizes();
    } else {
      dpath->clearCachedDir();
    }
  } else {
    dpath->clearCachedSize();
  }

  TF_SetStatus(status, TF_OK, "");
  return 0;
}

void DFS::dfsNewFile(dfs_path_t* dpath, File_Mode file_mode, int flags,
                     dfs_obj_t** obj, TF_Status* status) {
  int rc;
  dfs_obj_t* temp_obj;
  mode_t open_flags;

  rc = dfsLookUp(dpath, &temp_obj, status);
  if (rc) {
    if (TF_GetCode(status) != TF_NOT_FOUND) {
      return;
    }
    if (file_mode == READ) {
      return;
    }
    TF_SetStatus(status, TF_OK, "");
  }

  if (temp_obj != NULL && dfsIsDirectory(temp_obj)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    libdfs->dfs_release(temp_obj);
    return;
  }

  if (temp_obj != NULL && file_mode == READ) {
    *obj = temp_obj;
    return;
  }

  if (!rc && file_mode == WRITE) {
    rc = dfsDeleteObject(dpath, false, false, status);
    if (rc) {
      libdfs->dfs_release(temp_obj);
      return;
    }
  }

  open_flags = GetFlags(file_mode);

  dfs_obj_t* parent;
  mode_t parent_mode;
  rc = dfsFindParent(dpath, &parent, status);
  if (rc) {
    libdfs->dfs_release(temp_obj);
    return;
  }
  rc = libdfs->dfs_get_mode(parent, &parent_mode);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Cannot retrieve object mode");
    libdfs->dfs_release(temp_obj);
    return;
  }
  if (parent != NULL && !S_ISDIR(parent_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    libdfs->dfs_release(temp_obj);
    return;
  }

  std::string base_name = dpath->getBaseName();
  rc = libdfs->dfs_open(dpath->getFsys(), parent, base_name.c_str(), flags,
                        open_flags, 0, 0, NULL, obj);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Creating Writable File");
    libdfs->dfs_release(temp_obj);
    return;
  }
}

// Look up an object path and return its dfs_obj_t* in *obj.  If this routine
// returns zero then *obj is guaranteed to contain a valid dfs_obj_t* which
// must be released by the caller.  If non-zero, then *obj will be nullptr.  If
// the object happens to be a directory, a dup of the dfs_obj_t* will also be
// separately cached in the filesystem's directory cache.

int DFS::dfsLookUp(dfs_path_t* dpath, dfs_obj_t** obj, TF_Status* status) {
  *obj = NULL;
  int rc;

  // Check if the object path is for a directory we have seen before.

  dfs_obj_t* _obj = dpath->getCachedDir();
  if (_obj) {
    rc = libdfs->dfs_dup(dpath->getFsys(), _obj, O_RDWR, obj);
    if (rc) {
      TF_SetStatus(status, TF_INTERNAL, "dfs_dup() of open directory failed");
      return -1;
    }
    return 0;
  }

  if (dpath->isRoot()) {
    rc = libdfs->dfs_lookup(dpath->getFsys(), dpath->getRelPath().c_str(),
                            O_RDWR, &_obj, NULL, NULL);
  } else {
    dfs_obj_t* parent = NULL;
    rc = dfsFindParent(dpath, &parent, status);
    if (rc) return -1;

    dfs_t* fsys = dpath->getFsys();
    std::string basename = dpath->getBaseName();
    rc = libdfs->dfs_lookup_rel(fsys, parent, basename.c_str(), O_RDWR, &_obj,
                                NULL, NULL);
  }
  if (rc != 0) {
    if (rc == ENOENT) {
      TF_SetStatus(status, TF_NOT_FOUND, "");
    } else {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    }
    return -1;
  }

  if (!dfsIsDirectory(_obj)) {
    *obj = _obj;
    return 0;
  }

  // The object is a directory, so return the original dfs_obj_t* and store a
  // dup of the original in the filesystem's directory cache.

  rc = libdfs->dfs_dup(dpath->getFsys(), _obj, O_RDWR, obj);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "dfs_dup() of open directory failed");
    return -1;
  }
  dpath->setCachedDir(_obj);
  return 0;
}

// Given an object pathname, return the dfs_obj_t* for its parent directory.
// Cache the parent directory if not already cached.  The caller should not
// release the parent dfs_obj_t*.

int DFS::dfsFindParent(dfs_path_t* dpath, dfs_obj_t** obj, TF_Status* status) {
  *obj = NULL;
  int rc;

  dfs_path_t parent_dpath = *dpath;
  parent_dpath.setRelPath(dpath->getParentPath());

  dfs_obj_t* _obj = parent_dpath.getCachedDir();
  if (_obj) {
    *obj = _obj;
    return 0;
  }

  rc = libdfs->dfs_lookup(parent_dpath.getFsys(),
                          parent_dpath.getRelPath().c_str(), O_RDWR, &_obj,
                          NULL, NULL);
  if (rc) {
    if (rc == ENOENT) {
      TF_SetStatus(status, TF_NOT_FOUND, "");
    } else {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
    }
    return -1;
  }

  if (!dfsIsDirectory(_obj)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "parent object is not a directory");
    libdfs->dfs_release(_obj);
    return -1;
  }

  parent_dpath.setCachedDir(_obj);
  *obj = _obj;

  return 0;
}

int DFS::dfsCreateDir(dfs_path_t* dpath, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  dfs_obj_t* temp_obj;
  int rc;

  rc = dfsLookUp(dpath, &temp_obj, status);
  if (!rc) {
    if (dfsIsDirectory(temp_obj)) {
      libdfs->dfs_release(temp_obj);
      TF_SetStatus(status, TF_ALREADY_EXISTS, "");
      return 0;
    } else {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "");
      libdfs->dfs_release(temp_obj);
      return -1;
    }
  } else if (TF_GetCode(status) != TF_NOT_FOUND) {
    return -1;
  }

  TF_SetStatus(status, TF_OK, "");
  dfs_obj_t* parent;
  rc = dfsFindParent(dpath, &parent, status);
  if (rc) {
    return rc;
  }

  rc = libdfs->dfs_mkdir(dpath->getFsys(), parent, dpath->getBaseName().c_str(),
                         S_IWUSR | S_IRUSR | S_IXUSR, 0);
  if (rc) {
    TF_SetStatus(status, TF_INTERNAL, "Error Creating Directory");
  }

  return rc;
}

bool DFS::dfsIsDirectory(dfs_obj_t* obj) {
  if (obj == NULL) {
    return true;
  }
  mode_t mode;
  libdfs->dfs_get_mode(obj, &mode);
  if (S_ISDIR(mode)) {
    return true;
  }
  return false;
}

int DFS::dfsReadDir(dfs_t* daos_fs, dfs_obj_t* obj,
                    std::vector<std::string>& children) {
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
    return 0;
  }

  daos_handle_t poh;
  daos_pool_info_t info;
  rc = libdfs->daos_pool_connect2(pool_string.c_str(), NULL, DAOS_PC_RW, &poh,
                                  &info, NULL);
  if (rc == 0) {
    pool_info_t* po_inf = new pool_info_t();
    po_inf->poh = poh;
    pools[pool_string] = po_inf;
  }
  return rc;
}

int DFS::ConnectContainer(DFS* daos, std::string pool_string,
                          std::string cont_string, int allow_creation,
                          TF_Status* status) {
  int rc = 0;

  pool_info_t* po_inf = pools[pool_string];
  auto search = po_inf->containers.find(cont_string);
  if (search != po_inf->containers.end()) {
    return 0;
  }

  daos_handle_t coh;
  daos_cont_info_t info;
  rc = libdfs->daos_cont_open2(po_inf->poh, cont_string.c_str(), DAOS_COO_RW,
                               &coh, &info, NULL);
  if (rc == -DER_NONEXIST) {
    if (allow_creation) {
      rc = libdfs->dfs_cont_create_with_label(po_inf->poh, cont_string.c_str(),
                                              NULL, NULL, &coh, NULL);
    }
  }
  if (rc != 0) return rc;

  dfs_t* daos_fs;
  rc = libdfs->dfs_mount(po_inf->poh, coh, O_RDWR, &daos_fs);
  if (rc != 0) return rc;

  cont_info_t* co_inf = new cont_info_t();
  co_inf->coh = coh;
  co_inf->daos = daos;
  co_inf->daos_fs = daos_fs;
  co_inf->pool = pool_string;
  co_inf->cont = cont_string;

  po_inf->containers[cont_string] = co_inf;
  return 0;
}

int DFS::DisconnectPool(std::string pool_string) {
  int rc = 0;
  pool_info_t* po_inf = pools[pool_string];

  for (;;) {
    auto cont_it = po_inf->containers.cbegin();
    if (cont_it == po_inf->containers.cend()) {
      break;
    }
    rc = DisconnectContainer(pool_string, (*cont_it++).first);
    if (rc) return rc;
  }

  rc = libdfs->daos_pool_disconnect(po_inf->poh, NULL);
  if (rc == 0) {
    delete po_inf;
    pools.erase(pool_string);
  }
  return rc;
}

int DFS::DisconnectContainer(std::string pool_string, std::string cont_string) {
  int rc = 0;
  cont_info_t* co_inf = pools[pool_string]->containers[cont_string];

  if (co_inf->daos_fs) {
    rc = libdfs->dfs_umount(co_inf->daos_fs);
    if (rc) return rc;
    co_inf->daos_fs = nullptr;
  }

  rc = libdfs->daos_cont_close(co_inf->coh, nullptr);
  if (rc == 0) {
    delete co_inf;
    pools[pool_string]->containers.erase(cont_string);
  }
  return rc;
}

ReadBuffer::ReadBuffer(size_t aId, DFS* daos, daos_handle_t aEqh, size_t size)
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
  event->ev_error = daos->libdfs->dfs_read(daos_fs, file, &rsgl, buffer_offset,
                                           &read_size, event);
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

dfs_path_t::dfs_path_t(cont_info_t* cont_info, std::string rel_path)
    : cont_info(cont_info), rel_path(rel_path) {}

dfs_path_t& dfs_path_t::operator=(dfs_path_t other) {
  cont_info = other.cont_info;
  rel_path = other.rel_path;
  return *this;
}

std::string dfs_path_t::getFullPath(void) {
  std::string full_path =
      "/" + cont_info->pool + "/" + cont_info->cont + rel_path;
  return full_path;
}

std::string dfs_path_t::getRelPath(void) { return rel_path; }

std::string dfs_path_t::getParentPath(void) {
  if (rel_path == "/") {
    return rel_path;  // root is its own parent
  }

  std::string parent_path;
  size_t slash_pos = rel_path.rfind("/");
  if (slash_pos == 0) {
    parent_path = "/";
  } else {
    parent_path = rel_path.substr(0, slash_pos);
  }
  return parent_path;
}

std::string dfs_path_t::getBaseName(void) {
  size_t base_start = rel_path.rfind("/") + 1;
  std::string base_name = rel_path.substr(base_start);
  return base_name;
}

DFS* dfs_path_t::getDAOS(void) { return cont_info->daos; }

dfs_t* dfs_path_t::getFsys(void) { return cont_info->daos_fs; }

void dfs_path_t::setRelPath(std::string new_path) { rel_path = new_path; }

bool dfs_path_t::isRoot(void) { return (rel_path == "/"); }

dfs_obj_t* dfs_path_t::getCachedDir(void) {
  auto search = cont_info->dir_map.find(rel_path);
  if (search != cont_info->dir_map.end()) {
    return search->second;
  } else {
    return nullptr;
  }
}

void dfs_path_t::setCachedDir(dfs_obj_t* dir_obj) {
  cont_info->dir_map[rel_path] = dir_obj;
}

void dfs_path_t::clearCachedDir(void) { cont_info->dir_map.erase(rel_path); }

void dfs_path_t::clearFsysCachedDirs(void) {
  cont_info->daos->clearDirCache(cont_info->dir_map);
}

int dfs_path_t::getCachedSize(daos_size_t& size) {
  auto search = cont_info->size_map.find(rel_path);
  if (search != cont_info->size_map.end()) {
    size = search->second;
    return 0;
  } else {
    return -1;
  }
}

void dfs_path_t::setCachedSize(daos_size_t size) {
  cont_info->size_map[rel_path] = size;
}

void dfs_path_t::clearCachedSize(void) { cont_info->size_map.erase(rel_path); }

void dfs_path_t::clearFsysCachedSizes(void) {
  cont_info->daos->clearSizeCache(cont_info->size_map);
}

static void* LoadSharedLibrary(const char* library_filename,
                               TF_Status* status) {
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

static void* GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                  TF_Status* status) {
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

libDFS::~libDFS() {
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

void libDFS::LoadAndBindDaosLibs(TF_Status* status) {
#define LOAD_DFS_LIBRARY(handle, library_filename, status) \
  do {                                                     \
    handle = LoadSharedLibrary(library_filename, status);  \
    if (TF_GetCode(status) != TF_OK) return;               \
  } while (0);

  LOAD_DFS_LIBRARY(libdaos_handle_, "libdaos.so", status);
  LOAD_DFS_LIBRARY(libdfs_handle_, "libdfs.so", status);
  LOAD_DFS_LIBRARY(libduns_handle_, "libduns.so", status);

#undef LOAD_DFS_LIBRARY

#define BIND_DFS_FUNC(handle, function)             \
  do {                                              \
    BindFunc(handle, #function, &function, status); \
    if (TF_GetCode(status) != TF_OK) return;        \
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
  BIND_DFS_FUNC(libdfs_handle_, dfs_dup);
  BIND_DFS_FUNC(libdfs_handle_, dfs_get_mode);
  BIND_DFS_FUNC(libdfs_handle_, dfs_get_size);
  BIND_DFS_FUNC(libdfs_handle_, dfs_lookup);
  BIND_DFS_FUNC(libdfs_handle_, dfs_lookup_rel);
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
