/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <chrono>
#include <fstream>
#include <ostream>
#include <sstream>

#if defined(_MSC_VER)
#include <Windows.h>
#include <io.h>
#endif

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "blob/blob_client.h"
#include "logging.h"
#include "storage_account.h"
#include "storage_credential.h"
#include "storage_errno.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_io/core/plugins/file_system_plugins.h"

namespace tensorflow {
namespace io {
namespace az {
namespace {
// TODO: DO NOT use a hardcoded path
bool GetTmpFilename(std::string* filename) {
  if (!filename) {
    // return errors::Internal("'filename' cannot be nullptr.");
    return false;
  }
#ifndef _WIN32
  char buffer[] = "/tmp/az_blob_filesystem_XXXXXX";
  int fd = mkstemp(buffer);
  if (fd < 0) {
    // return errors::Internal("Failed to create a temporary file.");
    return false;
  }
#else
  char buffer[] = "/tmp/az_blob_filesystem_XXXXXX";
  char* ret = _mktemp(buffer);
  if (ret == nullptr) {
    // return errors::Internal("Failed to create a temporary file.");
    return false;
  }
#endif
  *filename = buffer;
  // return Status::OK();
  return true;
}

void ParseURI(const absl::string_view& fname, absl::string_view* scheme,
              absl::string_view* host, absl::string_view* path) {
  size_t scheme_chunk = fname.find("://");
  if (scheme_chunk == absl::string_view::npos) {
    return;
  }
  size_t host_chunk = fname.find("/", scheme_chunk + 3);
  if (host_chunk == absl::string_view::npos) {
    return;
  }
  *scheme = absl::string_view(fname).substr(0, scheme_chunk);
  *host = fname.substr(scheme_chunk + 3, host_chunk);
  *path = fname.substr(host_chunk, -1);
}

constexpr char kAzBlobEndpoint[] = ".blob.core.windows.net";

/// \brief Splits a Azure path to a account, container and object.
///
/// For example,
/// "az://account-name.blob.core.windows.net/container/path/to/file.txt" gets
/// split into "account-name", "container" and "path/to/file.txt".
void ParseAzBlobPath(const std::string& fname, bool empty_object_ok,
                     std::string* account, std::string* container,
                     std::string* object, TF_Status* status) {
  if (!account || !object) {
    TF_SetStatus(status, TF_INTERNAL, "account and object cannot be null");
    return;
  }
  absl::string_view scheme, accountp, objectp;
  ParseURI(fname, &scheme, &accountp, &objectp);
  if (scheme != "az") {
    std::string error_message = absl::StrCat(
        "Azure Blob Storage path doesn't start with 'az://': ", fname);
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }

  // Consume blob.core.windows.net if it exists
  absl::ConsumeSuffix(&accountp, kAzBlobEndpoint);

  if (accountp.empty() || accountp.compare(".") == 0) {
    std::string error_message = absl::StrCat(
        "Azure Blob Storage path doesn't contain a account name: ", fname);
    TF_SetStatus(status, TF_INVALID_ARGUMENT, error_message.c_str());
    return;
  }

  *account = std::string(accountp);

  absl::ConsumePrefix(&objectp, "/");

  auto pos = objectp.find('/');
  if (pos == std::string::npos) {
    *container = objectp.data();
    *object = "";
  } else {
    *container = std::string(objectp.substr(0, pos));
    *object = std::string(objectp.substr(pos + 1));
  }

  if (!empty_object_ok && object->empty()) {
    std::string error_message = absl::StrCat(
        "Azure Blob Storage path doesn't contain a object name: ", fname);
    TF_SetStatus(status, TF_INVALID_ARGUMENT, error_message.c_str());
    return;
  }

  TF_SetStatus(status, TF_OK, "");
  return;
}

std::string errno_to_string() {
  switch (errno) {
    /* common errors */
    case invalid_parameters:
      return "invalid_parameters";
    /* client level */
    case client_init_fail:
      return "client_init_fail";
    case client_already_init:
      return "client_already_init";
    case client_not_init:
      return "client_not_init";
    /* container level */
    case container_already_exists:
      return "container_already_exists";
    case container_not_exists:
      return "container_not_exists";
    case container_name_invalid:
      return "container_name_invalid";
    case container_create_fail:
      return "container_create_fail";
    case container_delete_fail:
      return "container_delete_fail";
    /* blob level */
    case blob_already_exists:
      return "blob__already_exists";
    case blob_not_exists:
      return "blob_not_exists";
    case blob_name_invalid:
      return "blob_name_invalid";
    case blob_delete_fail:
      return "blob_delete_fail";
    case blob_list_fail:
      return "blob_list_fail";
    case blob_copy_fail:
      return "blob_copy_fail";
    case blob_no_content_range:
      return "blob_no_content_range";
    /* unknown error */
    case unknown_error:
    default:
      return "unknown_error - " + std::to_string(errno);
  }
}

std::shared_ptr<azure::storage_lite::storage_credential> get_credential(
    const std::string& account) {
  const auto key = std::getenv("TF_AZURE_STORAGE_KEY");
  if (key != nullptr) {
    return std::make_shared<azure::storage_lite::shared_key_credential>(account,
                                                                        key);
  } else {
    return std::make_shared<azure::storage_lite::anonymous_credential>();
  }
}

azure::storage_lite::blob_client_wrapper CreateAzBlobClientWrapper(
    const std::string& account) {
  azure::storage_lite::logger::set_logger(
      [](azure::storage_lite::log_level level, const std::string& log_msg) {
        switch (level) {
          case azure::storage_lite::log_level::info:
            TF_Log(TF_INFO, log_msg.c_str());
            break;
          case azure::storage_lite::log_level::error:
          case azure::storage_lite::log_level::critical:
            TF_Log(TF_ERROR, log_msg.c_str());
            break;
          case azure::storage_lite::log_level::warn:
            TF_Log(TF_WARNING, log_msg.c_str());
            break;
          case azure::storage_lite::log_level::trace:
          case azure::storage_lite::log_level::debug:
          default:
            break;
        }
      });

  const auto use_dev_account = std::getenv("TF_AZURE_USE_DEV_STORAGE");
  if (use_dev_account != nullptr) {
    auto storage_account =
        azure::storage_lite::storage_account::development_storage_account();
    auto blob_client =
        std::make_shared<azure::storage_lite::blob_client>(storage_account, 10);
    azure::storage_lite::blob_client_wrapper blob_client_wrapper(blob_client);
    return blob_client_wrapper;
  }

  const auto use_http_env = std::getenv("TF_AZURE_STORAGE_USE_HTTP");
  const auto use_https = use_http_env == nullptr;
  const auto blob_endpoint_env = std::getenv("TF_AZURE_STORAGE_BLOB_ENDPOINT");
  const auto blob_endpoint =
      std::string(blob_endpoint_env ? blob_endpoint_env : "");

  auto credentials = get_credential(account);
  auto storage_account = std::make_shared<azure::storage_lite::storage_account>(
      account, credentials, use_https, blob_endpoint);
  auto blob_client =
      std::make_shared<azure::storage_lite::blob_client>(storage_account, 10);
  azure::storage_lite::blob_client_wrapper blob_client_wrapper(blob_client);

  return blob_client_wrapper;
}

void ListResources(const std::string& dir, const std::string& delimiter,
                   azure::storage_lite::blob_client_wrapper& blob_client,
                   std::vector<std::string>* results, TF_Status* status) {
  if (!results) {
    TF_SetStatus(status, TF_INTERNAL, "results cannot be null");
    return;
  }

  std::string account, container, object;
  ParseAzBlobPath(dir, true, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  std::string continuation_token;

  if (container.empty()) {
    std::vector<azure::storage_lite::list_containers_item> containers;
    do {
      auto list_containers_response =
          blob_client.list_containers_segmented("", continuation_token);
      if (errno != 0) {
        std::string error_message =
            absl::StrCat("Failed to get containers of account ", dir, " (",
                         errno_to_string(), ")");
        TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
        return;
      }

      containers.insert(containers.end(), list_containers_response.begin(),
                        list_containers_response.end());
    } while (!continuation_token.empty());

    std::transform(
        std::begin(containers), std::end(containers),
        std::back_inserter(*results),
        [](azure::storage_lite::list_containers_item list_container_item)
            -> std::string { return list_container_item.name; });

  } else {
    std::vector<azure::storage_lite::list_blobs_segmented_item> blobs;
    do {
      auto list_blobs_response = blob_client.list_blobs_segmented(
          container, delimiter, continuation_token, object);
      if (errno != 0) {
        std::string error_message = absl::StrCat("Failed to get blobs of ", dir,
                                                 " (", errno_to_string(), ")");
        TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
        return;
      }

      blobs.insert(blobs.end(), list_blobs_response.blobs.begin(),
                   list_blobs_response.blobs.end());

      continuation_token = list_blobs_response.next_marker;
    } while (!continuation_token.empty());

    results->reserve(blobs.size());
    std::transform(
        blobs.begin(), blobs.end(), std::back_inserter(*results),
        [](azure::storage_lite::list_blobs_segmented_item list_blob_item)
            -> std::string { return list_blob_item.name; });
  }

  TF_SetStatus(status, TF_OK, "");
}

class AzBlobRandomAccessFile {
 public:
  AzBlobRandomAccessFile(const std::string& account,
                         const std::string& container,
                         const std::string& object)
      : account_(account), container_(container), object_(object) {}
  ~AzBlobRandomAccessFile() {}
  int64_t Read(uint64_t offset, size_t n, char* buffer,
               TF_Status* status) const {
    // If n == 0, then return Status::OK()
    // otherwise, if bytes_read < n then return OutofRange
    if (n == 0) {
      TF_SetStatus(status, TF_OK, "");
      return 0;
    }
    auto blob_client = CreateAzBlobClientWrapper(account_);
    auto blob_property = blob_client.get_blob_property(container_, object_);
    if (errno != 0) {
      TF_SetStatus(status, TF_INTERNAL, "Failed to get properties");
      return 0;
    }
    int64_t file_size = blob_property.size;

    size_t bytes_to_read = n;
    if (offset >= file_size) {
      bytes_to_read = 0;
    } else if (offset + n > file_size) {
      bytes_to_read = file_size - offset;
    }

    if (bytes_to_read > 0) {
      std::ostringstream oss;
      // https://stackoverflow.com/a/12481580
#ifndef __APPLE__
      oss.rdbuf()->pubsetbuf(buffer, bytes_to_read);
#endif

      blob_client.download_blob_to_stream(container_, object_, offset,
                                          bytes_to_read, oss);
      if (errno != 0) {
        std::string error_message = absl::StrCat(
            "Failed to get contents of az://", account_, kAzBlobEndpoint, "/",
            container_, "/", object_, " (", errno_to_string(), ")");
        TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
        return 0;
      }

#ifdef __APPLE__
      auto blob_string = oss.str();
      if (buffer != nullptr) {
        std::copy(blob_string.begin(), blob_string.end(), buffer);
      }
      bytes_to_read = blob_string.size();
#endif
    }

    if (bytes_to_read < n) {
      TF_SetStatus(status, TF_OUT_OF_RANGE, "EOF reached");
      return bytes_to_read;
    }
    TF_SetStatus(status, TF_OK, "");
    return bytes_to_read;
  }

 private:
  std::string account_;
  std::string container_;
  std::string object_;
};

class AzBlobWritableFile {
 public:
  AzBlobWritableFile(const std::string& account, const std::string& container,
                     const std::string& object)
      : account_(account),
        container_(container),
        object_(object),
        sync_needed_(true) {
    if (GetTmpFilename(&tmp_content_filename_)) {
      outfile_.open(tmp_content_filename_,
                    std::ofstream::binary | std::ofstream::app);
    }
  }

  ~AzBlobWritableFile() {
    TF_Status* status = TF_NewStatus();
    Close(status);
    TF_DeleteStatus(status);
  }

  void Append(const char* buffer, size_t n, TF_Status* status) {
    if (!outfile_.is_open()) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION,
                   "The internal temporary file is not writable");
      return;
    }

    std::string data(buffer, n);
    sync_needed_ = true;
    outfile_ << data;
    if (!outfile_.good()) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Could not append to the internal temporary file");
      return;
    }
    TF_SetStatus(status, TF_OK, "");
  }
  void Sync(TF_Status* status) {
    if (!outfile_.is_open()) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION,
                   "The internal temporary file is not writable");
      return;
    }
    if (!sync_needed_) {
      TF_SetStatus(status, TF_OK, "");
      return;
    }
    outfile_.flush();
    if (!outfile_.good()) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Could not write to the internal temporary file");
      return;
    }

    auto blob_client = CreateAzBlobClientWrapper(account_);
    blob_client.upload_file_to_blob(tmp_content_filename_, container_, object_);
    if (errno != 0) {
      std::string error_message =
          absl::StrCat("Failed to upload to az://", account_, "/", container_,
                       "/", object_, " (", errno_to_string(), ")");
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }
    sync_needed_ = false;
    TF_SetStatus(status, TF_OK, "");
  }

  void Close(TF_Status* status) {
    if (outfile_.is_open()) {
      Sync(status);
      if (TF_GetCode(status) != TF_OK) {
        return;
      }
      outfile_.close();
      std::remove(tmp_content_filename_.c_str());
    }
    TF_SetStatus(status, TF_OK, "");
  }

 private:
  std::string account_;
  std::string container_;
  std::string object_;
  std::string tmp_content_filename_;
  std::ofstream outfile_;
  bool sync_needed_;  // whether there is buffered data that needs to be synced
};

#if 0
Status GetMatchingPaths(const std::string& pattern, std::vector<std::string>* results) {
  const std::string& fixed_prefix =
      pattern.substr(0, pattern.find_first_of("*?[\\"));

  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPathClass(fixed_prefix, true, &account, &container, &object));

  auto blob_client = CreateAzBlobClientWrapper(account);

  std::vector<std::string> blobs;
  TF_RETURN_IF_ERROR(ListResources(fixed_prefix, "", blob_client, &blobs));

  std::string container_path;
  if (pattern.find(kAzBlobEndpoint) != std::string::npos) {
    container_path =
        io::JoinPath("az://", account + kAzBlobEndpoint, container);
  } else {
    container_path = io::JoinPath("az://", account, container);
  }

  std::transform(std::begin(blobs), std::end(blobs), std::begin(blobs),
                 [&container_path](const std::string& path) {
                   return io::JoinPath(container_path, path);
                 });

  std::copy_if(std::begin(blobs), std::end(blobs), std::back_inserter(*results),
               [&pattern](const std::string& full_path) {
                 return Env::Default()->MatchPath(full_path, pattern);
               });

  return Status::OK();
}
#endif

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

static void Cleanup(TF_RandomAccessFile* file) {
  auto az_file = static_cast<AzBlobRandomAccessFile*>(file->plugin_file);
  delete az_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
  auto az_file = static_cast<AzBlobRandomAccessFile*>(file->plugin_file);
  return az_file->Read(offset, n, buffer, status);
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

static void Cleanup(TF_WritableFile* file) {
  auto az_file = static_cast<AzBlobWritableFile*>(file->plugin_file);
  delete az_file;
}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
  auto az_file = static_cast<AzBlobWritableFile*>(file->plugin_file);
  az_file->Append(buffer, n, status);
}

static int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Stat not implemented");
  return -1;
}

static void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto az_file = static_cast<AzBlobWritableFile*>(file->plugin_file);
  az_file->Sync(status);
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
  auto az_file = static_cast<AzBlobWritableFile*>(file->plugin_file);
  az_file->Sync(status);
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
  auto az_file = static_cast<AzBlobWritableFile*>(file->plugin_file);
  az_file->Close(status);
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
namespace tf_az_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
  std::string account, container, object;
  ParseAzBlobPath(path, false, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  file->plugin_file = new AzBlobRandomAccessFile(account, container, object);

  TF_SetStatus(status, TF_OK, "");
}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
  std::string account, container, object;
  ParseAzBlobPath(path, false, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  file->plugin_file = new AzBlobWritableFile(account, container, object);

  TF_SetStatus(status, TF_OK, "");
}

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status) {
  std::string account, container, object;
  ParseAzBlobPath(path, false, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  file->plugin_file = new AzBlobWritableFile(account, container, object);

  TF_SetStatus(status, TF_OK, "");
}

static void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                            const char* path,
                                            TF_ReadOnlyMemoryRegion* region,
                                            TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "NewReadOnlyMemoryRegionFromFile not implemented");
}

static void CreateDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  std::string account, container, object;
  ParseAzBlobPath(path, true, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  if (container.empty()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Cannot create storage accounts");
    return;
  }

  // Blob storage has virtual folders. We can make sure the container exists
  auto blob_client_wrapper = CreateAzBlobClientWrapper(account);

  if (blob_client_wrapper.container_exists(container)) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }

  blob_client_wrapper.create_container(container);
  if (errno != 0) {
    std::string error_message = absl::StrCat(
        "Failed to create directory ", path, " (", errno_to_string(), ")");
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

static void RecursivelyCreateDir(const TF_Filesystem* filesystem,
                                 const char* path, TF_Status* status) {
  CreateDir(filesystem, path, status);
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  std::string account, container, object;
  ParseAzBlobPath(path, false, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  auto blob_client = CreateAzBlobClientWrapper(account);

  blob_client.delete_blob(container, object);
  if (errno != 0) {
    std::string error_message = absl::StrCat(
        "Failed to delete ", path, ": ", errno, "(", errno_to_string(), ")");
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

static void DeleteDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  // Doesn't support file delete - call GetChildren (without delimiter) and then
  // loop and delete

  std::string account, container, object;
  ParseAzBlobPath(path, false, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  if (container.empty()) {
    // Don't allow deleting entire storage accout as we can't create them from
    // this file system
    TF_SetStatus(
        status, TF_FAILED_PRECONDITION,
        "Cannot delete storage account, limited to blobs or containers");
    return;
  }

  auto blob_client = CreateAzBlobClientWrapper(account);

  // Check container exists
  // Just pull out the first path component representing the container
  if (object.empty()) {
    blob_client.delete_container(container);
    if (errno != 0) {
      std::string error_message =
          absl::StrCat("Error deleting ", path, " (", errno_to_string(), ")");
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }
  } else {
    // Delete all blobs under dirname prefix
    std::vector<std::string> children;
    ListResources(path, "", blob_client, &children, status);
    if (TF_GetCode(status) != TF_OK) {
      return;
    }

    for (const auto& child : children) {
      blob_client.delete_blob(container, child);
      if (errno != 0) {
        std::string error_message = absl::StrCat("Failed to delete ", child,
                                                 " (", errno_to_string(), ")");
        TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
        return;
      }
    }
  }

  TF_SetStatus(status, TF_OK, "");
}

static void DeleteRecursively(const TF_Filesystem* filesystem, const char* path,
                              uint64_t* undeleted_files,
                              uint64_t* undeleted_dirs, TF_Status* status) {
  *undeleted_files = 0;
  *undeleted_dirs = 0;
  DeleteDir(filesystem, path, status);
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
  std::string src_account, src_container, src_object;
  ParseAzBlobPath(src, false, &src_account, &src_container, &src_object,
                  status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  std::string dst_account, dst_container, dst_object;
  ParseAzBlobPath(dst, false, &dst_account, &dst_container, &dst_object,
                  status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  if (src_account != dst_account) {
    std::string error_message =
        absl::StrCat("Couldn't rename ", src, " to ", dst,
                     ": moving files between accounts is not supported");
    TF_SetStatus(status, TF_UNIMPLEMENTED, error_message.c_str());
    return;
  }

  auto blob_client = CreateAzBlobClientWrapper(src_account);

  blob_client.start_copy(src_container, src_object, dst_container, dst_object);
  if (errno != 0) {
    std::string error_message =
        absl::StrCat("Failed to start rename from ", src, " to ", dst, " (",
                     errno_to_string(), ")");
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }

  // Wait until copy completes
  // Status can be success, pending, aborted or failed
  std::string copy_status;
  do {
    if (!copy_status.empty()) {
#if defined(_MSC_VER)
      Sleep(1000);
#else
      sleep(1);
#endif
    }
    const auto dst_blob_property =
        blob_client.get_blob_property(dst_container, dst_object);
    copy_status = dst_blob_property.copy_status;
  } while (copy_status.find("pending") == 0 && !copy_status.empty());

  if (copy_status.find("success") == std::string::npos) {
    std::string error_message =
        absl::StrCat("Process of renaming from ", src, " to ", dst,
                     " resulted in status of ", copy_status);
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }

  blob_client.delete_blob(src_container, src_object);
  if (errno != 0) {
    std::string error_message =
        absl::StrCat("Failed to get delete after copy of ", src, " (",
                     errno_to_string(), ")");
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }

  TF_SetStatus(status, TF_OK, "");
}

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {
  // 128KB copy buffer
  constexpr size_t kCopyFileBufferSize = 128 * 1024;

  std::string src_account, src_container, src_object;
  ParseAzBlobPath(src, false, &src_account, &src_container, &src_object,
                  status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  std::unique_ptr<AzBlobRandomAccessFile> src_file(
      new AzBlobRandomAccessFile(src_account, src_container, src_object));

  std::string dst_account, dst_container, dst_object;
  ParseAzBlobPath(dst, false, &dst_account, &dst_container, &dst_object,
                  status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  std::unique_ptr<AzBlobWritableFile> dst_file(
      new AzBlobWritableFile(dst_account, dst_container, dst_object));

  uint64_t offset = 0;
  std::unique_ptr<char[]> buffer(new char[kCopyFileBufferSize]);
  while (TF_GetCode(status) == TF_OK) {
    int64_t bytes_to_read =
        src_file->Read(offset, kCopyFileBufferSize, buffer.get(), status);
    if (!(TF_GetCode(status) == TF_OK ||
          TF_GetCode(status) == TF_OUT_OF_RANGE)) {
      return;
    }
    dst_file->Append(buffer.get(), bytes_to_read, status);
    if (TF_GetCode(status) != TF_OK) {
      return;
    }
    offset += bytes_to_read;
  }
  dst_file->Close(status);
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  std::string account, container, object;
  ParseAzBlobPath(path, false, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  auto blob_client = CreateAzBlobClientWrapper(account);
  auto blob_exists = blob_client.blob_exists(container, object);
  if (errno != 0) {
    std::string error_message = absl::StrCat(
        "Failed to check if ", path, " exists (", errno_to_string(), ")");
    TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
    return;
  }
  if (!blob_exists) {
    std::string error_message =
        absl::StrCat("The specified path ", path, " was not found");
    TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

static bool IsDirectory(const TF_Filesystem* filesystem, const char* path,
                        TF_Status* status) {
  // Should check that account and container exist and that fname isn't a file
  // Azure storage file system is virtual and is created with path compenents in
  // blobs name so no need to check further

  std::string account, container, object;
  ParseAzBlobPath(path, true, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return false;
  }

  auto blob_client = CreateAzBlobClientWrapper(account);

  if (container.empty()) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Currently account exists check is not implemented");
    return false;
    // bool is_account;

    // TF_RETURN_IF_ERROR(AccountExists(account, &is_account, blob_client));
    // return is_account ? Status::OK()
    //                   : errors::NotFound("The specified account az://",
    //                                      account, " was not found.");
  }

  auto container_exists = blob_client.container_exists(container);
  if (!container_exists) {
    std::string error_message =
        absl::StrCat("The specified folder ", path, " was not found");
    TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
    return false;
  }

  if (!object.empty()) {
    // Lastly check fname doesn't point to a file
    auto blob_exists = blob_client.blob_exists(container, object);
    if (blob_exists) {
      std::string error_message =
          absl::StrCat("The specified folder ", path, " is not a directory");
      TF_SetStatus(status, TF_FAILED_PRECONDITION, error_message.c_str());
      return false;
    }
  }
  TF_SetStatus(status, TF_OK, "");
  return true;
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
  using namespace std::chrono;

  std::string account, container, object;
  ParseAzBlobPath(path, false, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  auto blob_client = CreateAzBlobClientWrapper(account);

  if (IsDirectory(filesystem, path, status)) {
    stats->length = 0;
    stats->mtime_nsec = 0;
    stats->is_directory = true;
    return;
  }

  PathExists(filesystem, path, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  auto blob_property = blob_client.get_blob_property(container, object);
  if (errno != 0) {
    std::string error_message = absl::StrCat(
        "Failed to get file stats for ", path, " (", errno_to_string(), ")");
    TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
    return;
  }

  stats->length = blob_property.size;
  stats->mtime_nsec =
      duration_cast<nanoseconds>(seconds(blob_property.last_modified)).count();
  stats->is_directory = false;
  TF_SetStatus(status, TF_OK, "");
}

static int GetChildren(const TF_Filesystem* filesystem, const char* path,
                       char*** entries, TF_Status* status) {
  std::string account, container, object;
  ParseAzBlobPath(path, false, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return 0;
  }

  auto blob_client = CreateAzBlobClientWrapper(account);

  std::string continuation_token;
  if (container.empty()) {
    std::vector<std::string> result;
    // TODO: iterate while continuation_token isn't empty
    auto list_containers =
        blob_client.list_containers_segmented("", continuation_token, INT_MAX);
    std::transform(
        begin(list_containers), end(list_containers),
        std::back_inserter(result),
        [](azure::storage_lite::list_containers_item item) -> std::string {
          return item.name;
        });

    int num_entries = result.size();
    *entries = static_cast<char**>(
        plugin_memory_allocate(num_entries * sizeof((*entries)[0])));
    for (int i = 0; i < num_entries; i++) {
      (*entries)[i] = static_cast<char*>(
          plugin_memory_allocate(strlen(result[i].c_str()) + 1));
      memcpy((*entries)[i], result[i].c_str(), strlen(result[i].c_str()) + 1);
    }
    TF_SetStatus(status, TF_OK, "");
    return num_entries;
  }

  if (!object.empty() && object.back() != '/') {
    object += "/";
  }

  auto list_blobs = blob_client.list_blobs_segmented(
      container, "/", continuation_token, object);
  if (errno != 0) {
    std::string error_message = absl::StrCat("Failed to get child of ", path,
                                             " (", errno_to_string(), ")");
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return 0;
  }

  std::vector<std::string> result;
  auto blobs = list_blobs.blobs;
  result.reserve(blobs.size());
  std::transform(
      std::begin(blobs), std::end(blobs), std::back_inserter(result),
      [&object](azure::storage_lite::list_blobs_segmented_item list_blob_item)
          -> std::string {
        // Remove the prefix from the name
        auto blob_name = list_blob_item.name;
        blob_name.erase(0, object.size());
        // Remove the trailing slash from folders
        if (blob_name.back() == '/') {
          blob_name.pop_back();
        }
        return blob_name;
      });

  int num_entries = result.size();
  *entries = static_cast<char**>(
      plugin_memory_allocate(num_entries * sizeof((*entries)[0])));
  for (int i = 0; i < num_entries; i++) {
    (*entries)[i] = static_cast<char*>(
        plugin_memory_allocate(strlen(result[i].c_str()) + 1));
    memcpy((*entries)[i], result[i].c_str(), strlen(result[i].c_str()) + 1);
  }
  TF_SetStatus(status, TF_OK, "");
  return num_entries;
}

static int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                           TF_Status* status) {
  std::string account, container, object;
  ParseAzBlobPath(path, false, &account, &container, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return 0;
  }

  auto blob_client = CreateAzBlobClientWrapper(account);
  auto blob_property = blob_client.get_blob_property(container, object);
  if (errno != 0) {
    std::string error_message = absl::StrCat(
        "Failed to get properties of ", path, " (", errno_to_string(), ")");
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return 0;
  }

  TF_SetStatus(status, TF_OK, "");
  return blob_property.size;
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  return strdup(uri);
}

}  // namespace tf_az_filesystem

}  // namespace

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
  ops->writable_file_ops->flush = tf_writable_file::Flush;
  ops->writable_file_ops->sync = tf_writable_file::Sync;
  ops->writable_file_ops->close = tf_writable_file::Close;

  ops->read_only_memory_region_ops = static_cast<TF_ReadOnlyMemoryRegionOps*>(
      plugin_memory_allocate(TF_READ_ONLY_MEMORY_REGION_OPS_SIZE));
  ops->read_only_memory_region_ops->cleanup =
      tf_read_only_memory_region::Cleanup;
  ops->read_only_memory_region_ops->data = tf_read_only_memory_region::Data;
  ops->read_only_memory_region_ops->length = tf_read_only_memory_region::Length;

  ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
      plugin_memory_allocate(TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_az_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_az_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_az_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_az_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_az_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_az_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_az_filesystem::CreateDir;
  ops->filesystem_ops->recursively_create_dir =
      tf_az_filesystem::RecursivelyCreateDir;
  ops->filesystem_ops->delete_file = tf_az_filesystem::DeleteFile;
  ops->filesystem_ops->delete_recursively = tf_az_filesystem::DeleteRecursively;
  ops->filesystem_ops->delete_dir = tf_az_filesystem::DeleteDir;
  ops->filesystem_ops->copy_file = tf_az_filesystem::CopyFile;
  ops->filesystem_ops->rename_file = tf_az_filesystem::RenameFile;
  ops->filesystem_ops->path_exists = tf_az_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_az_filesystem::Stat;
  ops->filesystem_ops->is_directory = tf_az_filesystem::IsDirectory;
  ops->filesystem_ops->get_file_size = tf_az_filesystem::GetFileSize;
  ops->filesystem_ops->get_children = tf_az_filesystem::GetChildren;
  ops->filesystem_ops->translate_name = tf_az_filesystem::TranslateName;
}

}  // namespace az
}  // namespace io
}  // namespace tensorflow
