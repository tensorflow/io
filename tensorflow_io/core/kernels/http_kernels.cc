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

// TODO: Replace curl_http_request.h to not depends on TensorFlow's
// C++ implementation
#include "tensorflow/core/platform/cloud/curl_http_request.h"
#include "tensorflow_io/core/kernels/file_system_plugins.h"

namespace tensorflow {
namespace io {
namespace http {
namespace {

class HTTPRandomAccessFile {
 public:
  HTTPRandomAccessFile(const std::string& uri) : uri_(uri) {}
  ~HTTPRandomAccessFile() {}
  int64_t Read(uint64_t offset, size_t n, char* buffer,
               TF_Status* status) const {
    // If n == 0, then return Status::OK()
    // otherwise, if bytes_read < n then return OutofRange
    if (n == 0) {
      TF_SetStatus(status, TF_OK, "");
      return 0;
    }
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    request->SetUri(uri_);
    request->SetRange(offset, offset + n - 1);
    request->SetResultBufferDirect(buffer, n);
    // TODO: Replace Status with non C++ status
    Status s = request->Send();
    if (!s.ok()) {
      TF_SetStatus(status, TF_INTERNAL, s.error_message().c_str());
      return 0;
    }
    size_t bytes_to_read = request->GetResultBufferDirectBytesTransferred();
    if (bytes_to_read < n) {
      TF_SetStatus(status, TF_OUT_OF_RANGE, "EOF reached");
      return bytes_to_read;
    }
    TF_SetStatus(status, TF_OK, "");
    return bytes_to_read;
  }

 private:
  string uri_;

 public:
  static std::shared_ptr<HttpRequest::Factory> http_request_factory_;
};

std::shared_ptr<HttpRequest::Factory>
    HTTPRandomAccessFile::http_request_factory_ =
        std::make_shared<CurlHttpRequest::Factory>();

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

static void Cleanup(TF_RandomAccessFile* file) {
  auto http_file = static_cast<HTTPRandomAccessFile*>(file->plugin_file);
  delete http_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
  auto http_file = static_cast<HTTPRandomAccessFile*>(file->plugin_file);
  return http_file->Read(offset, n, buffer, status);
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

static void Cleanup(TF_WritableFile* file) {}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Append not implemented");
}

static int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Stat not implemented");
  return -1;
}

static void Flush(const TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Flush not implemented");
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Sync not implemented");
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Close not implemented");
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
namespace tf_azfs_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
  file->plugin_file = new HTTPRandomAccessFile(path);

  TF_SetStatus(status, TF_OK, "");
}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "NewWritableFile not implemented");
}

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "NewAppendableFile not implemented");
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
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CreateDir not implemented");
}

static void RecursivelyCreateDir(const TF_Filesystem* filesystem,
                                 const char* path, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "RecursivelyCreateDir not implemented");
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DeleteFile not implemented");
}

static void DeleteDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DeleteDir not implemented");
}

static void DeleteRecursively(const TF_Filesystem* filesystem, const char* path,
                              uint64_t* undeleted_files,
                              uint64_t* undeleted_dirs, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DeleteRecursively not implemented");
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RenameFile not implemented");
}

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CopyFile not implemented");
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
  std::unique_ptr<HttpRequest> request(
      HTTPRandomAccessFile::http_request_factory_->Create());
  request->SetUri(path);
  Status s = request->Send();
  if (!s.ok()) {
    TF_SetStatus(status, TF_INTERNAL, s.error_message().c_str());
    return;
  }
  string length_string = request->GetResponseHeader("Content-Length");
  if (length_string == "") {
    std::string error_message =
        absl::StrCat("unable to check the Content-Length of the url: ", path);
    TF_SetStatus(status, TF_INVALID_ARGUMENT, error_message.c_str());
    return;
  }
  int64 length = 0;
  if (!strings::safe_strto64(length_string, &length)) {
    std::string error_message =
        absl::StrCat("unable to parse the Content-Length of the url: ", path,
                     " [", length_string, "]");
    TF_SetStatus(status, TF_INVALID_ARGUMENT, error_message.c_str());
    return;
  }

  string last_modified_string = request->GetResponseHeader("Last-Modified");

  stats->length = length;
  stats->mtime_nsec = 0;
  stats->is_directory = false;
  TF_SetStatus(status, TF_OK, "");
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  TF_FileStatistics stats;
  Stat(filesystem, path, &stats, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  TF_SetStatus(status, TF_OK, "");
}

static bool IsDirectory(const TF_Filesystem* filesystem, const char* path,
                        TF_Status* status) {
  TF_FileStatistics stats;
  Stat(filesystem, path, &stats, status);
  if (TF_GetCode(status) != TF_OK) {
    return false;
  }

  TF_SetStatus(status, TF_OK, "");
  return stats.is_directory;
}

static int GetChildren(const TF_Filesystem* filesystem, const char* path,
                       char*** entries, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "GetChildren not implemented");
  return 0;
}

static int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                           TF_Status* status) {
  TF_FileStatistics stats;
  Stat(filesystem, path, &stats, status);
  if (TF_GetCode(status) != TF_OK) {
    return 0;
  }

  TF_SetStatus(status, TF_OK, "");
  return stats.length;
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  return strdup(uri);
}

}  // namespace tf_azfs_filesystem

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
  ops->filesystem_ops->init = tf_azfs_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_azfs_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_azfs_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_azfs_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_azfs_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_azfs_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_azfs_filesystem::CreateDir;
  ops->filesystem_ops->recursively_create_dir =
      tf_azfs_filesystem::RecursivelyCreateDir;
  ops->filesystem_ops->delete_file = tf_azfs_filesystem::DeleteFile;
  ops->filesystem_ops->delete_recursively =
      tf_azfs_filesystem::DeleteRecursively;
  ops->filesystem_ops->delete_dir = tf_azfs_filesystem::DeleteDir;
  ops->filesystem_ops->copy_file = tf_azfs_filesystem::CopyFile;
  ops->filesystem_ops->rename_file = tf_azfs_filesystem::RenameFile;
  ops->filesystem_ops->path_exists = tf_azfs_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_azfs_filesystem::Stat;
  ops->filesystem_ops->is_directory = tf_azfs_filesystem::IsDirectory;
  ops->filesystem_ops->get_file_size = tf_azfs_filesystem::GetFileSize;
  ops->filesystem_ops->get_children = tf_azfs_filesystem::GetChildren;
  ops->filesystem_ops->translate_name = tf_azfs_filesystem::TranslateName;
}

}  // namespace http
}  // namespace io
}  // namespace tensorflow
