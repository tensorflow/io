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
#include "tensorflow_io/core/kernels/azfs_kernels.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"

#include "blob/blob_client.h"
#include "logging.h"
#include "storage_account.h"
#include "storage_credential.h"
#include "storage_errno.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <ostream>
#include <sstream>

namespace tensorflow {
namespace io {
namespace {
// TODO: DO NOT use a hardcoded path
Status GetTmpFilename(std::string *filename) {
  if (!filename) {
    return errors::Internal("'filename' cannot be nullptr.");
  }
#ifndef _WIN32
  char buffer[] = "/tmp/az_blob_filesystem_XXXXXX";
  int fd = mkstemp(buffer);
  if (fd < 0) {
    return errors::Internal("Failed to create a temporary file.");
  }
#else
  char buffer[] = "/tmp/az_blob_filesystem_XXXXXX";
  char *ret = _mktemp(buffer);
  if (ret == nullptr) {
    return errors::Internal("Failed to create a temporary file.");
  }
#endif
  *filename = buffer;
  return Status::OK();
}

constexpr char kAzBlobEndpoint[] = ".blob.core.windows.net";

/// \brief Splits a Azure path to a account, container and object.
///
/// For example,
/// "az://account-name.blob.core.windows.net/container/path/to/file.txt" gets
/// split into "account-name", "container" and "path/to/file.txt".
Status ParseAzBlobPath(StringPiece fname, bool empty_object_ok,
                       std::string *account, std::string *container,
                       std::string *object) {
  if (!account || !object) {
    return errors::Internal("account and object cannot be null.");
  }
  StringPiece scheme, accountp, objectp;
  io::ParseURI(fname, &scheme, &accountp, &objectp);
  if (scheme != "az") {
    return errors::InvalidArgument(
        "Azure Blob Storage path doesn't start with 'az://': ", fname);
  }

  // Consume blob.core.windows.net if it exists
  absl::ConsumeSuffix(&accountp, kAzBlobEndpoint);

  if (accountp.empty() || accountp.compare(".") == 0) {
    return errors::InvalidArgument(
        "Azure Blob Storage path doesn't contain a account name: ", fname);
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

  return Status::OK();
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
    case blob__already_exists:
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
    const std::string &account) {
  const auto key = std::getenv("TF_AZURE_STORAGE_KEY");
  if (key != nullptr) {
    return std::make_shared<azure::storage_lite::shared_key_credential>(account,
                                                                        key);
  } else {
    return std::make_shared<azure::storage_lite::anonymous_credential>();
  }
}

azure::storage_lite::blob_client_wrapper CreateAzBlobClientWrapper(
    const std::string &account) {
  azure::storage_lite::logger::set_logger(
      [](azure::storage_lite::log_level level, const std::string &log_msg) {
        switch (level) {
          case azure::storage_lite::log_level::info:
            _TF_LOG_INFO << log_msg;
            break;
          case azure::storage_lite::log_level::error:
          case azure::storage_lite::log_level::critical:
            _TF_LOG_ERROR << log_msg;
            break;
          case azure::storage_lite::log_level::warn:
            _TF_LOG_WARNING << log_msg;
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
  const auto blob_endpoint =
      std::string(std::getenv("TF_AZURE_STORAGE_BLOB_ENDPOINT") ?: "");

  auto credentials = get_credential(account);
  auto storage_account = std::make_shared<azure::storage_lite::storage_account>(
      account, credentials, use_https, blob_endpoint);
  auto blob_client =
      std::make_shared<azure::storage_lite::blob_client>(storage_account, 10);
  azure::storage_lite::blob_client_wrapper blob_client_wrapper(blob_client);

  return blob_client_wrapper;
}

class AzBlobRandomAccessFile : public RandomAccessFile {
 public:
  AzBlobRandomAccessFile(const std::string &account,
                         const std::string &container,
                         const std::string &object)
      : account_(account), container_(container), object_(object) {}
  ~AzBlobRandomAccessFile() {}
  Status Read(uint64 offset, size_t n, StringPiece *result,
              char *scratch) const override {
    // If n == 0, then return Status::OK()
    // otherwise, if bytes_read < n then return OutofRange
    if (n == 0) {
      *result = StringPiece("", 0);
      return Status::OK();
    }
    auto blob_client = CreateAzBlobClientWrapper(account_);
    auto blob_property = blob_client.get_blob_property(container_, object_);
    if (errno != 0) {
      return errors::Internal("Failed to get properties");
    }
    int64 file_size = blob_property.size;

    size_t bytes_to_read = n;
    if (offset >= file_size) {
      bytes_to_read = 0;
    } else if (offset + n > file_size) {
      bytes_to_read = file_size - offset;
    }

    if (bytes_to_read == 0) {
      *result = StringPiece("", 0);
    } else {
      std::ostringstream oss;
      // https://stackoverflow.com/a/12481580
#ifndef __APPLE__
      oss.rdbuf()->pubsetbuf(scratch, bytes_to_read);
#endif

      blob_client.download_blob_to_stream(container_, object_, offset,
                                          bytes_to_read, oss);
      if (errno != 0) {
        *result = StringPiece("", 0);
        return errors::Internal("Failed to get contents of az://", account_,
                                kAzBlobEndpoint, "/", container_, "/", object_,
                                " (", errno_to_string(), ")");
      }

#ifndef __APPLE__
      *result = StringPiece(scratch, bytes_to_read);
#else
      auto blob_string = oss.str();
      if (scratch != nullptr) {
        std::copy(blob_string.begin(), blob_string.end(), scratch);
      }
      *result = StringPiece(blob_string);
#endif
    }
    if (bytes_to_read < n) {
      return errors::OutOfRange("EOF reached");
    }

    return Status::OK();
  }

 private:
  std::string account_;
  std::string container_;
  std::string object_;
};

class AzBlobWritableFile : public WritableFile {
 public:
  AzBlobWritableFile(const std::string &account, const std::string &container,
                     const std::string &object)
      : account_(account),
        container_(container),
        object_(object),
        sync_needed_(true) {
    if (GetTmpFilename(&tmp_content_filename_).ok()) {
      outfile_.open(tmp_content_filename_,
                    std::ofstream::binary | std::ofstream::app);
    }
  }

  ~AzBlobWritableFile() { Close().IgnoreError(); }

  Status Append(StringPiece data) override {
    TF_RETURN_IF_ERROR(CheckWritable());
    sync_needed_ = true;
    outfile_ << data;
    if (!outfile_.good()) {
      return errors::Internal(
          "Could not append to the internal temporary file.");
    }
    return Status::OK();
  }

  Status Close() override {
    if (outfile_.is_open()) {
      TF_RETURN_IF_ERROR(Sync());
      outfile_.close();
      std::remove(tmp_content_filename_.c_str());
    }
    return Status::OK();
  }

  Status Flush() override { return Sync(); }
  Status Sync() override {
    TF_RETURN_IF_ERROR(CheckWritable());
    if (!sync_needed_) {
      return Status::OK();
    }
    const auto status = SyncImpl();
    if (status.ok()) {
      sync_needed_ = false;
    }
    return status;
  }

 private:
  Status SyncImpl() {
    outfile_.flush();
    if (!outfile_.good()) {
      return errors::Internal(
          "Could not write to the internal temporary file.");
    }

    auto blob_client = CreateAzBlobClientWrapper(account_);
    blob_client.upload_file_to_blob(tmp_content_filename_, container_, object_);
    if (errno != 0) {
      return errors::Internal("Failed to upload to az://", account_, "/",
                              container_, "/", object_, " (", errno_to_string(),
                              ")");
    }

    return Status::OK();
  }

  Status CheckWritable() const {
    if (!outfile_.is_open()) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    return Status::OK();
  }

  std::string account_;
  std::string container_;
  std::string object_;
  std::string tmp_content_filename_;
  std::ofstream outfile_;
  bool sync_needed_;  // whether there is buffered data that needs to be synced
};

class AzBlobReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  AzBlobReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {}
  const void *data() override { return reinterpret_cast<void *>(data_.get()); }
  uint64 length() override { return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

}  // namespace

Status AzBlobFileSystem::NewRandomAccessFile(
    const std::string &filename, std::unique_ptr<RandomAccessFile> *result) {
  string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(filename, false, &account, &container, &object));
  result->reset(new AzBlobRandomAccessFile(account, container, object));
  return Status::OK();
}

Status AzBlobFileSystem::NewWritableFile(
    const std::string &fname, std::unique_ptr<WritableFile> *result) {
  string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(fname, false, &account, &container, &object));
  result->reset(new AzBlobWritableFile(account, container, object));
  return Status::OK();
}

Status AzBlobFileSystem::NewAppendableFile(
    const std::string &fname, std::unique_ptr<WritableFile> *result) {
  string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(fname, false, &account, &container, &object));
  result->reset(new AzBlobWritableFile(account, container, object));
  return Status::OK();
}

Status AzBlobFileSystem::NewReadOnlyMemoryRegionFromFile(
    const std::string &filename,
    std::unique_ptr<ReadOnlyMemoryRegion> *result) {
  uint64 size;
  TF_RETURN_IF_ERROR(GetFileSize(filename, &size));
  std::unique_ptr<char[]> data(new char[size]);

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(filename, &file));

  StringPiece piece;
  TF_RETURN_IF_ERROR(file->Read(0, size, &piece, data.get()));

  result->reset(new AzBlobReadOnlyMemoryRegion(std::move(data), size));
  return Status::OK();
}

Status AzBlobFileSystem::FileExists(const std::string &fname) {
  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(fname, false, &account, &container, &object));
  auto blob_client = CreateAzBlobClientWrapper(account);
  auto blob_exists = blob_client.blob_exists(container, object);
  if (errno != 0) {
    return errors::Internal("Failed to check if ", fname, " exists (",
                            errno_to_string(), ")");
  }
  if (!blob_exists) {
    return errors::NotFound("The specified path ", fname, " was not found.");
  }
  return Status::OK();
}

Status AzBlobFileSystem::Stat(const std::string &fname, FileStatistics *stat) {
  using namespace std::chrono;

  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(fname, false, &account, &container, &object));
  auto blob_client = CreateAzBlobClientWrapper(account);

  if (IsDirectory(fname).ok()) {
    *stat = FileStatistics(0, 0, true);
    return Status::OK();
  }

  if (!FileExists(fname).ok()) {
    return errors::NotFound("The specified object ", fname, " was not found");
  }

  auto blob_property = blob_client.get_blob_property(container, object);
  if (errno != 0) {
    return errors::Internal("Failed to get file stats for ", fname, " (",
                            errno_to_string(), ")");
  }

  FileStatistics fs;
  fs.length = blob_property.size;
  fs.mtime_nsec =
      duration_cast<nanoseconds>(seconds(blob_property.last_modified)).count();

  *stat = std::move(fs);

  return Status::OK();
}

Status AzBlobFileSystem::GetChildren(const std::string &dir,
                                     std::vector<std::string> *result) {
  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(dir, false, &account, &container, &object));
  auto blob_client = CreateAzBlobClientWrapper(account);

  std::string continuation_token;
  if (container.empty()) {
    // TODO: iterate while continuation_token isn't empty
    auto list_containers =
        blob_client.list_containers_segmented("", continuation_token, INT_MAX);
    std::transform(
        begin(list_containers), end(list_containers),
        std::back_inserter(*result),
        [](azure::storage_lite::list_containers_item item) -> std::string {
          return item.name;
        });
    return Status::OK();
  }

  if (!object.empty() && object.back() != '/') {
    object += "/";
  }

  auto list_blobs = blob_client.list_blobs_segmented(
      container, "/", continuation_token, object);
  if (errno != 0) {
    return errors::Internal("Failed to get child of ", dir, " (",
                            errno_to_string(), ")");
  }

  auto blobs = list_blobs.blobs;
  result->reserve(blobs.size());
  std::transform(
      std::begin(blobs), std::end(blobs), std::back_inserter(*result),
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

  return Status::OK();
}

Status AzBlobFileSystem::GetMatchingPaths(const std::string &pattern,
                                          std::vector<std::string> *results) {
  const std::string &fixed_prefix =
      pattern.substr(0, pattern.find_first_of("*?[\\"));

  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(fixed_prefix, true, &account, &container, &object));

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
                 [&container_path](const std::string &path) {
                   return io::JoinPath(container_path, path);
                 });

  std::copy_if(std::begin(blobs), std::end(blobs), std::back_inserter(*results),
               [&pattern](const std::string &full_path) {
                 return Env::Default()->MatchPath(full_path, pattern);
               });

  return Status::OK();
}

Status AzBlobFileSystem::DeleteFile(const std::string &fname) {
  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(fname, false, &account, &container, &object));
  auto blob_client = CreateAzBlobClientWrapper(account);

  blob_client.delete_blob(container, object);
  if (errno != 0) {
    return errors::Internal("Failed to delete ", fname, " (", errno_to_string(),
                            ")");
  }

  return Status::OK();
}

Status AzBlobFileSystem::CreateDir(const std::string &dirname) {
  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(dirname, true, &account, &container, &object));
  if (container.empty()) {
    return errors::FailedPrecondition("Cannot create storage accounts");
  }

  // Blob storage has virtual folders. We can make sure the container exists
  auto blob_client_wrapper = CreateAzBlobClientWrapper(account);

  if (blob_client_wrapper.container_exists(container)) {
    return Status::OK();
  }

  blob_client_wrapper.create_container(container);
  if (errno != 0) {
    return errors::Internal("Failed to create directory ", dirname, " (",
                            errno_to_string(), ")");
  }
  return Status::OK();
}

Status AzBlobFileSystem::DeleteDir(const std::string &dirname) {
  // Doesn't support file delete - call GetChildren (without delimiter) and then
  // loop and delete

  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(dirname, false, &account, &container, &object));
  if (container.empty()) {
    // Don't allow deleting entire storage accout as we can't create them from
    // this file system
    return errors::FailedPrecondition(
        "Cannot delete storage account, limited to blobs or containers");
  }

  auto blob_client = CreateAzBlobClientWrapper(account);

  // Check container exists
  // Just pull out the first path component representing the container
  if (object.empty()) {
    blob_client.delete_container(container);
    if (errno != 0) {
      return errors::Internal("Error deleting ", dirname, " (",
                              errno_to_string(), ")");
    }
  } else {
    // Delete all blobs under dirname prefix
    std::vector<std::string> children;
    TF_RETURN_IF_ERROR(ListResources(dirname, "", blob_client, &children));

    for (const auto &child : children) {
      blob_client.delete_blob(container, child);
      if (errno != 0) {
        return errors::Internal("Failed to delete ", child, " (",
                                errno_to_string(), ")");
      }
    }
  }

  return Status::OK();
}

Status AzBlobFileSystem::GetFileSize(const std::string &fname,
                                     uint64 *file_size) {
  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(fname, false, &account, &container, &object));

  auto blob_client = CreateAzBlobClientWrapper(account);
  auto blob_property = blob_client.get_blob_property(container, object);
  if (errno != 0) {
    return errors::Internal("Failed to get properties of ", fname, " (",
                            errno_to_string(), ")");
  }
  *file_size = blob_property.size;

  return Status::OK();
}

Status AzBlobFileSystem::RenameFile(const std::string &src,
                                    const std::string &target) {
  std::string src_account, src_container, src_object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(src, false, &src_account, &src_container, &src_object));
  std::string target_account, target_container, target_object;
  TF_RETURN_IF_ERROR(ParseAzBlobPath(target, false, &target_account,
                                     &target_container, &target_object));

  if (src_account != target_account) {
    return errors::Unimplemented(
        "Couldn't rename ", src, " to ", target,
        ": moving files between accounts is not supported.");
  }

  auto blob_client = CreateAzBlobClientWrapper(src_account);

  blob_client.start_copy(src_container, src_object, target_container,
                         target_object);
  if (errno != 0) {
    return errors::Internal("Failed to start rename from ", src, " to ", target,
                            " (", errno_to_string(), ")");
  }

  // Wait until copy completes
  // Status can be success, pending, aborted or failed
  std::string copy_status;
  do {
    const auto target_blob_property =
        blob_client.get_blob_property(target_container, target_object);
    copy_status = target_blob_property.copy_status;
  } while (copy_status.find("pending") == 0 && !copy_status.empty());

  if (copy_status.find("success") == std::string::npos) {
    return errors::Internal("Process of renaming resulted in status of ",
                            copy_status, " when renaming ", src, " to ",
                            target);
  }

  blob_client.delete_blob(src_container, src_object);
  if (errno != 0) {
    return errors::Internal("Failed to get delete after copy of ", src, " (",
                            errno_to_string(), ")");
  }

  return Status::OK();
}

Status AzBlobFileSystem::RecursivelyCreateDir(const string &dirname) {
  return CreateDir(dirname);
}

Status AzBlobFileSystem::IsDirectory(const std::string &fname) {
  // Should check that account and container exist and that fname isn't a file
  // Azure storage file system is virtual and is created with path compenents in
  // blobs name so no need to check further

  std::string account, container, object;
  TF_RETURN_IF_ERROR(
      ParseAzBlobPath(fname, true, &account, &container, &object));

  auto blob_client = CreateAzBlobClientWrapper(account);

  if (container.empty()) {
    return errors::Unimplemented(
        "Currently account exists check is not implemented");
    // bool is_account;

    // TF_RETURN_IF_ERROR(AccountExists(account, &is_account, blob_client));
    // return is_account ? Status::OK()
    //                   : errors::NotFound("The specified account az://",
    //                                      account, " was not found.");
  }

  auto container_exists = blob_client.container_exists(container);
  if (!container_exists) {
    return errors::NotFound("The specified folder ", fname, " was not found");
  }

  if (!object.empty()) {
    // Lastly check fname doesn't point to a file
    auto blob_exists = blob_client.blob_exists(container, object);
    if (blob_exists) {
      return errors::FailedPrecondition("The specified path ", fname,
                                        " is not a directory.");
    }
  }

  // If account & container exist & fname isn't a file, with virtual directories
  // we say that fname is a directory
  return Status::OK();
}

Status AzBlobFileSystem::DeleteRecursively(const std::string &dirname,
                                           int64 *undeleted_files,
                                           int64 *undeleted_dirs) {
  TF_RETURN_IF_ERROR(DeleteDir(dirname));
  *undeleted_dirs = 0;
  *undeleted_files = 0;

  return Status::OK();
}

void AzBlobFileSystem::FlushCaches() {}

Status AzBlobFileSystem::ListResources(
    const std::string &dir, const std::string &delimiter,
    azure::storage_lite::blob_client_wrapper &blob_client,
    std::vector<std::string> *results) const {
  if (!results) {
    return errors::Internal("results cannot be null");
  }

  std::string account, container, object;
  TF_RETURN_IF_ERROR(ParseAzBlobPath(dir, true, &account, &container, &object));

  std::string continuation_token;

  if (container.empty()) {
    std::vector<azure::storage_lite::list_containers_item> containers;
    do {
      auto list_containers_response =
          blob_client.list_containers_segmented("", continuation_token);
      if (errno != 0) {
        return errors::Internal("Failed to get containers of account ", dir,
                                " (", errno_to_string(), ")");
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
        return errors::Internal("Failed to get blobs of ", dir, " (",
                                errno_to_string(), ")");
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

  return Status::OK();
}
namespace {
REGISTER_FILE_SYSTEM("az", io::AzBlobFileSystem);
}  // namespace
}  // namespace io
}  // namespace tensorflow
