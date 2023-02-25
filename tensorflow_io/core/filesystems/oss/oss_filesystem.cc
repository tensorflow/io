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
#include "tensorflow_io/core/filesystems/oss/oss_filesystem.h"

#include <pwd.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "aos_string.h"
#include "oss_define.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow_io/core/filesystems/filesystem_plugins.h"

namespace tensorflow {
namespace io {
namespace oss {

constexpr char kOSSCredentialsDefaultFile[] = ".osscredentials";
constexpr char kOSSCredentialsFileEnvKey[] = "OSS_CREDENTIALS";
constexpr char kOSSCredentialsSection[] = "OSSCredentials";
constexpr char kOSSCredentialsHostKey[] = "host";
constexpr char kOSSCredentialsAccessIdKey[] = "accessid";
constexpr char kOSSCredentialsAccesskeyKey[] = "accesskey";
constexpr char kOSSAccessIdKey[] = "id";
constexpr char kOSSAccessKeyKey[] = "key";
constexpr char kOSSHostKey[] = "host";
constexpr char kDelim[] = "/";
static char oss_user_agent[256] = "";

void oss_initialize_with_throwable() {
  if (aos_http_io_initialize(NULL, 0) != AOSE_OK) {
    throw std::exception();
  }
  std::string user_agent = aos_default_http_transport_options->user_agent;
  user_agent += std::string(", TensorFlow I/O");
  if (user_agent.size() < 256) {
    strncpy(oss_user_agent, user_agent.c_str(), user_agent.size());
    aos_default_http_transport_options->user_agent = oss_user_agent;
  }
}

Status oss_initialize() {
  static std::once_flag initFlag;
  try {
    std::call_once(initFlag, [] { oss_initialize_with_throwable(); });
  } catch (...) {
    LOG(FATAL) << "can not init OSS connection";
    return errors::Internal("can not init OSS connection");
  }

  return OkStatus();
}

void oss_error_message(aos_status_s* status, std::string* msg) {
  *msg = status->req_id;
  if (aos_status_is_ok(status)) {
    return;
  }

  msg->append(" ");
  msg->append(std::to_string(status->code));

  if (status->code == 404) {
    msg->append(" object not exists!");
    return;
  }

  if (status->error_msg) {
    msg->append(" ");
    msg->append(status->error_msg);
    return;
  }
}

class OSSConnection {
 public:
  OSSConnection(const std::string& endPoint, const std::string& accessKey,
                const std::string& accessKeySecret) {
    aos_pool_create(&_pool, NULL);
    _options = oss_request_options_create(_pool);
    _options->config = oss_config_create(_options->pool);
    aos_str_set(&_options->config->endpoint, endPoint.c_str());
    aos_str_set(&_options->config->access_key_id, accessKey.c_str());
    aos_str_set(&_options->config->access_key_secret, accessKeySecret.c_str());
    _options->config->is_cname = 0;
    _options->ctl = aos_http_controller_create(_options->pool, 0);
  }

  ~OSSConnection() {
    if (NULL != _pool) {
      aos_pool_destroy(_pool);
    }
  }

  oss_request_options_t* getRequestOptions() { return _options; }

  aos_pool_t* getPool() { return _pool; }

 private:
  aos_pool_t* _pool = NULL;
  oss_request_options_t* _options = NULL;
};

class OSSRandomAccessFile : public RandomAccessFile {
 public:
  OSSRandomAccessFile(const std::string& endPoint, const std::string& accessKey,
                      const std::string& accessKeySecret,
                      const std::string& bucket, const std::string& object,
                      size_t read_ahead_bytes, size_t file_length)
      : shost(endPoint),
        sak(accessKey),
        ssk(accessKeySecret),
        sbucket(bucket),
        sobject(object),
        total_file_length_(file_length) {
    read_ahead_bytes_ = std::min(read_ahead_bytes, file_length);
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    // offset is 0 based, so last offset should be
    // just before total_file_length_
    if (offset >= total_file_length_) {
      return errors::OutOfRange("EOF reached, ", offset,
                                " is read out of file length ",
                                total_file_length_);
    }

    if (offset + n > total_file_length_) {
      n = total_file_length_ - offset;
    }

    VLOG(1) << "read " << sobject << " from " << offset << " to " << offset + n;

    mutex_lock lock(mu_);
    const bool range_start_included = offset >= buffer_start_offset_;
    const bool range_end_included =
        offset + n <= buffer_start_offset_ + buffer_size_;
    if (range_start_included && range_end_included) {
      // The requested range can be filled from the buffer.
      const size_t offset_in_buffer =
          std::min<uint64>(offset - buffer_start_offset_, buffer_size_);
      const auto copy_size = std::min(n, buffer_size_ - offset_in_buffer);
      VLOG(1) << "read from buffer " << offset_in_buffer << " to "
              << offset_in_buffer + copy_size << " total " << buffer_size_;
      std::copy(buffer_.begin() + offset_in_buffer,
                buffer_.begin() + offset_in_buffer + copy_size, scratch);
      *result = StringPiece(scratch, copy_size);
    } else {
      // Update the buffer content based on the new requested range.
      const size_t desired_buffer_size =
          std::min(n + read_ahead_bytes_, total_file_length_);
      if (n > buffer_.capacity() ||
          desired_buffer_size > 2 * buffer_.capacity()) {
        // Re-allocate only if buffer capacity increased significantly.
        VLOG(1) << "reserve buffer to " << desired_buffer_size;
        buffer_.reserve(desired_buffer_size);
      }

      buffer_start_offset_ = offset;
      VLOG(1) << "load buffer" << buffer_start_offset_;
      TF_RETURN_IF_ERROR(LoadBufferFromOSS(desired_buffer_size));

      // Set the results.
      memcpy(scratch, buffer_.data(), std::min(buffer_size_, n));
      *result = StringPiece(scratch, std::min(buffer_size_, n));
    }

    if (result->size() < n) {
      // This is not an error per se. The RandomAccessFile interface expects
      // that Read returns OutOfRange if fewer bytes were read than requested.
      return errors::OutOfRange("EOF reached, ", result->size(),
                                " bytes were read out of ", n,
                                " bytes requested.");
    }
    return OkStatus();
  }

 private:
  /// A helper function to actually read the data from OSS. This function loads
  /// buffer_ from OSS based on its current capacity.
  Status LoadBufferFromOSS(size_t desired_buffer_size) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    size_t range_start = buffer_start_offset_;
    size_t range_end = buffer_start_offset_ + std::min(buffer_.capacity() - 1,
                                                       desired_buffer_size - 1);
    range_end = std::min(range_end, total_file_length_ - 1);

    OSSConnection conn(shost, sak, ssk);
    aos_pool_t* _pool = conn.getPool();
    oss_request_options_t* _options = conn.getRequestOptions();
    aos_string_t bucket_;
    aos_string_t object_;
    aos_table_t* headers_;
    aos_list_t tmp_buffer;
    aos_table_t* resp_headers;

    aos_list_init(&tmp_buffer);
    aos_str_set(&_options->config->endpoint, shost.c_str());
    aos_str_set(&_options->config->access_key_id, sak.c_str());
    aos_str_set(&_options->config->access_key_secret, ssk.c_str());
    _options->config->is_cname = 0;
    _options->ctl = aos_http_controller_create(_options->pool, 0);
    aos_str_set(&bucket_, sbucket.c_str());
    aos_str_set(&object_, sobject.c_str());
    headers_ = aos_table_make(_pool, 1);

    std::string range("bytes=");
    range.append(std::to_string(range_start))
        .append("-")
        .append(std::to_string(range_end));
    apr_table_set(headers_, "Range", range.c_str());
    VLOG(1) << "read from OSS with " << range.c_str();

    aos_status_t* s =
        oss_get_object_to_buffer(_options, &bucket_, &object_, headers_, NULL,
                                 &tmp_buffer, &resp_headers);
    if (aos_status_is_ok(s)) {
      aos_buf_t* content = NULL;
      int64_t size = 0;
      int64_t pos = 0;
      buffer_.clear();
      buffer_size_ = 0;

      // copy data to local buffer
      aos_list_for_each_entry(aos_buf_t, content, &tmp_buffer, node) {
        size = aos_buf_size(content);
        std::copy(content->pos, content->pos + size, buffer_.begin() + pos);
        pos += size;
      }
      buffer_size_ = pos;
      return OkStatus();
    } else {
      string msg;
      oss_error_message(s, &msg);
      VLOG(0) << "read " << sobject << " failed, errMsg: " << msg;
      return errors::Internal("read failed: ", sobject, " errMsg: ", msg);
    }
  }

  std::string shost;
  std::string sak;
  std::string ssk;
  std::string sbucket;
  std::string sobject;
  const size_t total_file_length_;
  size_t read_ahead_bytes_;

  mutable mutex mu_;
  mutable std::vector<char> buffer_ TF_GUARDED_BY(mu_);
  // The original file offset of the first byte in the buffer.
  mutable size_t buffer_start_offset_ TF_GUARDED_BY(mu_) = 0;
  mutable size_t buffer_size_ TF_GUARDED_BY(mu_) = 0;
};

class OSSReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  OSSReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {}
  const void* data() override { return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override { return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

class OSSWritableFile : public WritableFile {
 public:
  OSSWritableFile(const std::string& endPoint, const std::string& accessKey,
                  const std::string& accessKeySecret, const std::string& bucket,
                  const std::string& object, size_t part_size)
      : shost(endPoint),
        sak(accessKey),
        ssk(accessKeySecret),
        sbucket(bucket),
        sobject(object),
        part_size_(part_size),
        is_closed_(false),
        part_number_(1) {
    InitAprPool();
  }

  ~OSSWritableFile() { ReleaseAprPool(); }

  Status Append(StringPiece data) override {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(_CheckClosed());
    InitAprPool();
    if (CurrentBufferLength() >= part_size_) {
      TF_RETURN_IF_ERROR(_FlushInternal());
    }

    aos_buf_t* tmp_buf = aos_create_buf(pool_, data.size() + 1);
    aos_buf_append_string(pool_, tmp_buf, data.data(), data.size());
    aos_list_add_tail(&tmp_buf->node, &buffer_);
    return OkStatus();
  }

  Status Close() override {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(_CheckClosed());
    InitAprPool();
    TF_RETURN_IF_ERROR(_FlushInternal());
    aos_table_t* complete_headers = NULL;
    aos_table_t* resp_headers = NULL;
    aos_status_t* status = NULL;
    oss_list_upload_part_params_t* params = NULL;
    aos_list_t complete_part_list;
    oss_list_part_content_t* part_content = NULL;
    oss_complete_part_content_t* complete_part_content = NULL;
    aos_string_t upload_id;
    aos_str_set(&upload_id, upload_id_.c_str());

    params = oss_create_list_upload_part_params(pool_);
    aos_list_init(&complete_part_list);
    status = oss_list_upload_part(options_, &bucket_, &object_, &upload_id,
                                  params, &resp_headers);

    if (!aos_status_is_ok(status)) {
      string msg;
      oss_error_message(status, &msg);
      VLOG(0) << "List multipart " << sobject << " failed, errMsg: " << msg;
      return errors::Internal("List multipart failed: ", sobject,
                              " errMsg: ", msg);
    }

    aos_list_for_each_entry(oss_list_part_content_t, part_content,
                            &params->part_list, node) {
      complete_part_content = oss_create_complete_part_content(pool_);
      aos_str_set(&complete_part_content->part_number,
                  part_content->part_number.data);
      aos_str_set(&complete_part_content->etag, part_content->etag.data);
      aos_list_add_tail(&complete_part_content->node, &complete_part_list);
    }

    status = oss_complete_multipart_upload(options_, &bucket_, &object_,
                                           &upload_id, &complete_part_list,
                                           complete_headers, &resp_headers);

    if (!aos_status_is_ok(status)) {
      string msg;
      oss_error_message(status, &msg);
      VLOG(0) << "Complete multipart " << sobject << " failed, errMsg: " << msg;
      return errors::Internal("Complete multipart failed: ", sobject,
                              " errMsg: ", msg);
    }

    is_closed_ = true;
    return OkStatus();
  }

  Status Flush() override {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(_CheckClosed());
    if (CurrentBufferLength() >= part_size_) {
      InitAprPool();
      TF_RETURN_IF_ERROR(_FlushInternal());
    }

    return OkStatus();
  }

  Status Sync() override { return Flush(); }

 private:
  void InitAprPool() {
    if (NULL == pool_) {
      aos_pool_create(&pool_, NULL);
      options_ = oss_request_options_create(pool_);
      options_->config = oss_config_create(options_->pool);
      aos_str_set(&options_->config->endpoint, shost.c_str());
      aos_str_set(&options_->config->access_key_id, sak.c_str());
      aos_str_set(&options_->config->access_key_secret, ssk.c_str());
      options_->config->is_cname = 0;
      options_->ctl = aos_http_controller_create(options_->pool, 0);

      aos_str_set(&bucket_, sbucket.c_str());
      aos_str_set(&object_, sobject.c_str());

      headers_ = aos_table_make(pool_, 1);
      aos_list_init(&buffer_);
    }
  }

  void ReleaseAprPool() {
    if (NULL != pool_) {
      aos_pool_destroy(pool_);
      pool_ = NULL;
    }
  }

  Status _InitMultiUpload() {
    if (upload_id_.empty()) {
      aos_string_t uploadId;
      aos_status_t* status = NULL;
      aos_table_t* resp_headers = NULL;

      InitAprPool();
      status = oss_init_multipart_upload(options_, &bucket_, &object_,
                                         &uploadId, headers_, &resp_headers);

      if (!aos_status_is_ok(status)) {
        string msg;
        oss_error_message(status, &msg);
        VLOG(0) << "Init multipart upload " << sobject
                << " failed, errMsg: " << msg;
        return errors::Unavailable("Init multipart upload failed: ", sobject,
                                   " errMsg: ", msg);
      }

      upload_id_ = uploadId.data;
    }

    return OkStatus();
  }

  Status _FlushInternal() {
    aos_table_t* resp_headers = NULL;
    aos_status_s* status = NULL;
    aos_string_t uploadId;
    if (CurrentBufferLength() > 0) {
      _InitMultiUpload();

      aos_str_set(&uploadId, upload_id_.c_str());
      status =
          oss_upload_part_from_buffer(options_, &bucket_, &object_, &uploadId,
                                      part_number_, &buffer_, &resp_headers);

      if (!aos_status_is_ok(status)) {
        string msg;
        oss_error_message(status, &msg);
        VLOG(0) << "Upload multipart " << sobject << " failed, errMsg: " << msg;
        return errors::Internal("Upload multipart failed: ", sobject,
                                " errMsg: ", msg);
      }

      VLOG(1) << " upload " << sobject << " with part" << part_number_
              << " succ";
      part_number_++;
      ReleaseAprPool();
      InitAprPool();
    }
    return OkStatus();
  }

  const size_t CurrentBufferLength() { return aos_buf_list_len(&buffer_); }

  Status _CheckClosed() {
    if (is_closed_) {
      return errors::Internal("Already closed.");
    }

    return OkStatus();
  }

  std::string shost;
  std::string sak;
  std::string ssk;
  std::string sbucket;
  std::string sobject;
  size_t part_size_;

  aos_pool_t* pool_ = NULL;
  oss_request_options_t* options_ = NULL;
  aos_string_t bucket_;
  aos_string_t object_;
  aos_table_t* headers_ = NULL;
  aos_list_t buffer_;
  std::string upload_id_;

  bool is_closed_;
  mutex mu_;
  int64_t part_number_;
};

OSSFileSystem::OSSFileSystem() {}

// Splits a oss path to endpoint bucket object and token
// For example
// "oss://bucket-name\x01id=accessid\x02key=accesskey\x02host=endpoint/path/to/file.txt"
Status OSSFileSystem::_ParseOSSURIPath(const StringPiece fname,
                                       std::string& bucket, std::string& object,
                                       std::string& host,
                                       std::string& access_id,
                                       std::string& access_key) {
  StringPiece scheme, bucketp, remaining;
  io::ParseURI(fname, &scheme, &bucketp, &remaining);

  if (scheme != "oss") {
    return errors::InvalidArgument("OSS path does not start with 'oss://':",
                                   fname);
  }

  str_util::ConsumePrefix(&remaining, kDelim);
  object = string(remaining);

  std::string bucketDelim = "?";
  std::string accessDelim = "&";
  if (bucketp.find('\x01') != StringPiece::npos) {
    bucketDelim = "\x01";
    accessDelim = "\x02";
  }

  // contains id, key, host information
  size_t pos = bucketp.find(bucketDelim);
  bucket = string(bucketp.substr(0, pos));
  StringPiece access_info = bucketp.substr(pos + 1);
  std::vector<std::string> access_infos =
      str_util::Split(access_info, accessDelim);
  for (const auto& key_value : access_infos) {
    StringPiece data(key_value);
    size_t pos = data.find('=');
    if (pos == StringPiece::npos) {
      return errors::InvalidArgument("OSS path access info faied: ", fname,
                                     " info:", key_value);
    }
    StringPiece key = data.substr(0, pos);
    StringPiece value = data.substr(pos + 1);
    if (str_util::StartsWith(key, kOSSAccessIdKey)) {
      access_id = string(value);
    } else if (str_util::StartsWith(key, kOSSAccessKeyKey)) {
      access_key = string(value);
    } else if (str_util::StartsWith(key, kOSSHostKey)) {
      host = string(value);
    } else {
      return errors::InvalidArgument("OSS path access info faied: ", fname,
                                     " unkown info:", key_value);
    }
  }

  if (bucket.empty()) {
    return errors::InvalidArgument("OSS path does not contain a bucket name:",
                                   fname);
  }

  if (access_id.empty() || access_key.empty() || host.empty()) {
    return errors::InvalidArgument(
        "OSS path does not contain valid access info:", fname);
  }

  VLOG(1) << "bucket: " << bucket << ",access_id: " << access_id
          << ",access_key: " << access_key << ",host: " << host;

  return OkStatus();
}

Status OSSFileSystem::NewRandomAccessFile(
    const std::string& filename, std::unique_ptr<RandomAccessFile>* result) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(filename, bucket, object, host, access_id, access_key));
  TF_FileStatistics stat;
  OSSConnection conn(host, access_id, access_key);
  TF_RETURN_IF_ERROR(_RetrieveObjectMetadata(
      conn.getPool(), conn.getRequestOptions(), bucket, object, &stat));
  result->reset(new OSSRandomAccessFile(host, access_id, access_key, bucket,
                                        object, read_ahead_bytes_,
                                        stat.length));
  return OkStatus();
}

Status OSSFileSystem::NewWritableFile(const std::string& fname,
                                      std::unique_ptr<WritableFile>* result) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(fname, bucket, object, host, access_id, access_key));

  result->reset(new OSSWritableFile(host, access_id, access_key, bucket, object,
                                    upload_part_bytes_));
  return OkStatus();
}

Status OSSFileSystem::NewAppendableFile(const std::string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  return errors::Unimplemented(
      "Does not support appendable file in OSSFileSystem");
}

Status OSSFileSystem::NewReadOnlyMemoryRegionFromFile(
    const std::string& filename,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  uint64 size;
  TF_RETURN_IF_ERROR(GetFileSize(filename, &size));
  std::unique_ptr<char[]> data(new char[size]);

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(filename, &file));

  StringPiece piece;
  TF_RETURN_IF_ERROR(file->Read(0, size, &piece, data.get()));

  result->reset(new OSSReadOnlyMemoryRegion(std::move(data), size));
  return OkStatus();
}

Status OSSFileSystem::FileExists(const std::string& fname) {
  TF_FileStatistics stat;
  if (Stat(fname, &stat).ok()) {
    return OkStatus();
  } else {
    return errors::NotFound(fname, " does not exists");
  }
}

// For GetChildren , we should not return prefix
Status OSSFileSystem::_ListObjects(
    aos_pool_t* pool, const oss_request_options_t* options,
    const std::string& bucket, const std::string& key,
    std::vector<std::string>* result, bool return_all, bool return_full_path,
    bool should_remove_suffix, bool recursive, int max_ret_per_iterator) {
  aos_string_t bucket_;
  aos_status_t* s = NULL;
  oss_list_object_params_t* params = NULL;
  oss_list_object_content_t* content = NULL;
  const char* next_marker = "";

  aos_str_set(&bucket_, bucket.c_str());
  params = oss_create_list_object_params(pool);
  params->max_ret = max_ret_per_iterator;
  aos_str_set(&params->prefix, key.c_str());
  aos_str_set(&params->marker, next_marker);
  if (!recursive) {
    aos_str_set(&params->delimiter, "/");
  }

  do {
    s = oss_list_object(options, &bucket_, params, NULL);
    if (!aos_status_is_ok(s)) {
      string msg;
      oss_error_message(s, &msg);
      VLOG(0) << "cam not list object " << key << " errMsg: " << msg;
      return errors::NotFound("can not list object:", key, " errMsg: ", msg);
    }

    aos_list_for_each_entry(oss_list_object_content_t, content,
                            &params->object_list, node) {
      int path_length = content->key.len;
      if (should_remove_suffix && path_length > 0 &&
          content->key.data[content->key.len - 1] == '/') {
        path_length = content->key.len - 1;
      }
      if (return_full_path) {
        string child(content->key.data, 0, path_length);
        result->push_back(child);
      } else {
        int prefix_len = (key.length() > 0 && key.at(key.length() - 1) != '/')
                             ? key.length() + 1
                             : key.length();
        // remove prefix for GetChildren
        if (content->key.len > prefix_len) {
          string child(content->key.data + prefix_len, 0,
                       path_length - prefix_len);
          result->push_back(child);
        }
      }
    }

    aos_list_for_each_entry(oss_list_object_content_t, content,
                            &params->common_prefix_list, node) {
      int path_length = content->key.len;
      if (should_remove_suffix && path_length > 0 &&
          content->key.data[content->key.len - 1] == '/') {
        path_length = content->key.len - 1;
      }
      if (return_full_path) {
        string child(content->key.data, 0, path_length);
        result->push_back(child);
      } else {
        int prefix_len = (key.length() > 0 && key.at(key.length() - 1) != '/')
                             ? key.length() + 1
                             : key.length();
        // remove prefix for GetChildren
        if (content->key.len > prefix_len) {
          string child(content->key.data + prefix_len, 0,
                       path_length - prefix_len);
          result->push_back(child);
        }
      }
    }

    next_marker = apr_psprintf(pool, "%.*s", params->next_marker.len,
                               params->next_marker.data);

    aos_str_set(&params->marker, next_marker);
    aos_list_init(&params->object_list);
    aos_list_init(&params->common_prefix_list);
  } while (params->truncated == AOS_TRUE && return_all);

  return OkStatus();
}

Status OSSFileSystem::_StatInternal(aos_pool_t* pool,
                                    const oss_request_options_t* options,
                                    const std::string& bucket,
                                    const std::string& object,
                                    TF_FileStatistics* stat) {
  Status s = _RetrieveObjectMetadata(pool, options, bucket, object, stat);
  if (s.ok()) {
    VLOG(1) << "RetrieveObjectMetadata for object: " << object
            << " file success";
    return s;
  }

  // add suffix
  std::string objectName = object + kDelim;
  s = _RetrieveObjectMetadata(pool, options, bucket, objectName, stat);
  if (s.ok()) {
    VLOG(1) << "RetrieveObjectMetadata for object: " << objectName
            << " directory success";
    stat->is_directory = true;
    return s;
  }

  // check list if it has children
  std::vector<std::string> listing;
  s = _ListObjects(pool, options, bucket, object, &listing, true, false, false,
                   true, 10);

  if (s == OkStatus() && !listing.empty()) {
    if (str_util::EndsWith(object, "/")) {
      stat->is_directory = true;
    }
    stat->length = 0;
    VLOG(1) << "RetrieveObjectMetadata for object: " << object
            << " get children success";
    return s;
  }

  VLOG(1) << "_StatInternal for object: " << object
          << ", failed with bucket: " << bucket;
  return errors::NotFound("can not find ", object);
}

Status OSSFileSystem::_RetrieveObjectMetadata(
    aos_pool_t* pool, const oss_request_options_t* options,
    const std::string& bucket, const std::string& object,
    TF_FileStatistics* stat) {
  aos_string_t oss_bucket;
  aos_string_t oss_object;
  aos_table_t* headers = NULL;
  aos_table_t* resp_headers = NULL;
  aos_status_t* status = NULL;
  char* content_length_str = NULL;
  char* object_date_str = NULL;

  if (object.empty()) {  // root always exists
    stat->is_directory = true;
    stat->length = 0;
    return OkStatus();
  }

  aos_str_set(&oss_bucket, bucket.c_str());
  aos_str_set(&oss_object, object.c_str());
  headers = aos_table_make(pool, 0);

  status = oss_head_object(options, &oss_bucket, &oss_object, headers,
                           &resp_headers);
  if (aos_status_is_ok(status)) {
    content_length_str = (char*)apr_table_get(resp_headers, OSS_CONTENT_LENGTH);
    if (content_length_str != NULL) {
      stat->length = static_cast<int64>(atoll(content_length_str));
      VLOG(1) << "_RetrieveObjectMetadata object: " << object
              << " , with length: " << stat->length;
    }

    object_date_str = (char*)apr_table_get(resp_headers, OSS_DATE);
    if (object_date_str != NULL) {
      // the time is GMT Date, format like below
      // Date: Fri, 24 Feb 2012 07:32:52 GMT
      std::tm tm = {};
      strptime(object_date_str, "%a, %d %b %Y %H:%M:%S", &tm);
      stat->mtime_nsec = static_cast<int64>(mktime(&tm) * 1000) * 1e9;

      VLOG(1) << "_RetrieveObjectMetadata object: " << object
              << " , with time: " << stat->mtime_nsec;
    } else {
      VLOG(0) << "find " << object << " with no datestr";
      return errors::NotFound("find ", object, " with no datestr");
    }

    if (object[object.length() - 1] == '/') {
      stat->is_directory = true;
    } else {
      stat->is_directory = false;
    }

    return OkStatus();
  } else {
    string msg;
    oss_error_message(status, &msg);
    VLOG(1) << "can not find object: " << object << ", with bucket: " << bucket
            << ", errMsg: " << msg;
    return errors::NotFound("can not find ", object, " errMsg: ", msg);
  }
}

Status OSSFileSystem::Stat(const std::string& fname, TF_FileStatistics* stat) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(fname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* ossOptions = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();

  return _StatInternal(pool, ossOptions, bucket, object, stat);
}

Status OSSFileSystem::GetChildren(const std::string& dir,
                                  std::vector<std::string>* result) {
  result->clear();
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dir, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  if (!object.empty() && object.back() != '/') object.push_back('/');
  return _ListObjects(pool, oss_options, bucket, object, result, true, false,
                      true, false, 1000);
}

Status OSSFileSystem::_DeleteObjectInternal(
    const oss_request_options_t* options, const std::string& bucket,
    const std::string& object) {
  aos_string_t bucket_;
  aos_string_t object_;
  aos_table_t* resp_headers = NULL;
  aos_status_t* s = NULL;

  aos_str_set(&bucket_, bucket.c_str());
  aos_str_set(&object_, object.c_str());

  s = oss_delete_object(options, &bucket_, &object_, &resp_headers);
  if (!aos_status_is_ok(s)) {
    string msg;
    oss_error_message(s, &msg);
    VLOG(0) << "delete " << object << " failed, errMsg: " << msg;
    return errors::Internal("delete failed: ", object, " errMsg: ", msg);
  }

  return OkStatus();
}

Status OSSFileSystem::DeleteFile(const std::string& fname) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(fname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();

  return _DeleteObjectInternal(oss_options, bucket, object);
}

Status OSSFileSystem::CreateDir(const std::string& dirname) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dirname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* ossOptions = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  StringPiece dirs(object);

  std::vector<std::string> splitPaths =
      str_util::Split(dirs, '/', str_util::SkipEmpty());
  if (splitPaths.size() < 2) {
    return _CreateDirInternal(pool, ossOptions, bucket, object);
  }

  TF_FileStatistics stat;
  StringPiece parent = io::Dirname(dirs);

  if (!_StatInternal(pool, ossOptions, bucket, string(parent), &stat).ok()) {
    VLOG(0) << "CreateDir() failed with bucket: " << bucket
            << ", parent: " << parent;
    return errors::Internal("parent does not exists: ", parent);
  }

  if (!stat.is_directory) {
    return errors::Internal("can not mkdir because parent is a file: ", parent);
  }

  TF_RETURN_IF_ERROR(_CreateDirInternal(pool, ossOptions, bucket, object));
  return OkStatus();
}

Status OSSFileSystem::RecursivelyCreateDir(const string& dirname) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dirname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* ossOptions = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  StringPiece dirs(object);

  std::vector<std::string> splitPaths =
      str_util::Split(dirs, '/', str_util::SkipEmpty());
  if (splitPaths.size() < 2) {
    return _CreateDirInternal(pool, ossOptions, bucket, object);
  }

  std::string dir = "";
  for (auto path : splitPaths) {
    dir.append(path + kDelim);

    if (!_CreateDirInternal(pool, ossOptions, bucket, dir).ok()) {
      VLOG(0) << "create dir failed with bucket: " << bucket
              << ", dir: " << dir;
      return errors::Internal("create dir failed: ", dir);
    }
  }

  return OkStatus();
}

Status OSSFileSystem::_CreateDirInternal(aos_pool_t* pool,
                                         const oss_request_options_t* options,
                                         const std::string& bucket,
                                         const std::string& dirname) {
  TF_FileStatistics stat;
  if (_RetrieveObjectMetadata(pool, options, bucket, dirname, &stat).ok()) {
    if (!stat.is_directory) {
      VLOG(0) << "object already exists as a file: " << dirname;
      return errors::AlreadyExists("object already exists as a file: ",
                                   dirname);
    } else {
      return OkStatus();
    }
  }
  std::string object = dirname;
  if (dirname.at(dirname.length() - 1) != '/') {
    object += '/';
  }

  aos_status_t* s;
  aos_table_t* headers;
  aos_table_t* resp_headers;
  aos_string_t bucket_;
  aos_string_t object_;
  const char* data = "";
  aos_list_t buffer;
  aos_buf_t* content;

  aos_str_set(&bucket_, bucket.c_str());
  aos_str_set(&object_, object.c_str());
  headers = aos_table_make(pool, 0);

  aos_list_init(&buffer);
  content = aos_buf_pack(options->pool, data, strlen(data));
  aos_list_add_tail(&content->node, &buffer);
  s = oss_put_object_from_buffer(options, &bucket_, &object_, &buffer, headers,
                                 &resp_headers);

  if (aos_status_is_ok(s)) {
    return OkStatus();
  } else {
    string msg;
    oss_error_message(s, &msg);
    VLOG(1) << "mkdir " << dirname << " failed, errMsg: " << msg;
    return errors::Internal("mkdir failed: ", dirname, " errMsg: ", msg);
  }
}

Status OSSFileSystem::DeleteDir(const std::string& dirname) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dirname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  std::vector<std::string> children;
  Status s;

  s = _ListObjects(pool, oss_options, bucket, object, &children, true, false,
                   false, 10);
  if (s.ok() && !children.empty()) {
    return errors::FailedPrecondition("Cannot delete a non-empty directory.");
  }

  s = _DeleteObjectInternal(oss_options, bucket, object);

  if (s.ok()) {
    return s;
  }

  // Maybe should add slash
  return _DeleteObjectInternal(oss_options, bucket, object.append(kDelim));
}

Status OSSFileSystem::GetFileSize(const std::string& fname, uint64* file_size) {
  TF_FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(fname, &stat));
  *file_size = stat.length;
  return OkStatus();
}

Status OSSFileSystem::RenameFile(const std::string& src,
                                 const std::string& target) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string sobject, sbucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(src, sbucket, sobject, host, access_id, access_key));
  std::string dobject, dbucket;
  std::string dhost, daccess_id, daccess_key;
  TF_RETURN_IF_ERROR(_ParseOSSURIPath(target, dbucket, dobject, dhost,
                                      daccess_id, daccess_key));

  if (host != dhost || access_id != daccess_id || access_key != daccess_key) {
    VLOG(0) << "rename " << src << " to " << target << " failed, with errMsg: "
            << " source oss cluster does not match dest oss cluster";
    return errors::Internal(
        "rename ", src, " to ", target, " failed, errMsg: ",
        "source oss cluster does not match dest oss cluster");
  }

  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();

  aos_status_t* resp_status;
  aos_string_t source_bucket;
  aos_string_t source_object;
  aos_string_t dest_bucket;
  aos_string_t dest_object;

  aos_str_set(&source_bucket, sbucket.c_str());
  aos_str_set(&dest_bucket, dbucket.c_str());

  Status status = IsDirectory(src);
  if (status.ok()) {
    if (!str_util::EndsWith(sobject, "/")) {
      sobject += "/";
    }
    if (!str_util::EndsWith(dobject, "/")) {
      dobject += "/";
    }
    std::vector<std::string> childPaths;
    _ListObjects(pool, oss_options, sbucket, sobject, &childPaths, true, false,
                 false, true, 1000);
    for (const auto& child : childPaths) {
      std::string tmp_sobject = sobject + child;
      std::string tmp_dobject = dobject + child;

      aos_str_set(&source_object, tmp_sobject.c_str());
      aos_str_set(&dest_object, tmp_dobject.c_str());

      resp_status = _CopyFileInternal(oss_options, pool, source_bucket,
                                      source_object, dest_bucket, dest_object);
      if (!aos_status_is_ok(resp_status)) {
        string msg;
        oss_error_message(resp_status, &msg);
        VLOG(0) << "rename " << src << " to " << target
                << " failed, with specific file:  " << tmp_sobject
                << ", with errMsg: " << msg;
        return errors::Internal("rename ", src, " to ", target,
                                " failed, errMsg: ", msg);
      }
      _DeleteObjectInternal(oss_options, sbucket, tmp_sobject);
    }
  }

  aos_str_set(&source_object, sobject.c_str());
  aos_str_set(&dest_object, dobject.c_str());
  resp_status = _CopyFileInternal(oss_options, pool, source_bucket,
                                  source_object, dest_bucket, dest_object);
  if (!aos_status_is_ok(resp_status)) {
    string msg;
    oss_error_message(resp_status, &msg);
    VLOG(0) << "rename " << src << " to " << target
            << " failed, errMsg: " << msg;
    return errors::Internal("rename ", src, " to ", target,
                            " failed, errMsg: ", msg);
  }

  return _DeleteObjectInternal(oss_options, sbucket, sobject);
}

aos_status_t* OSSFileSystem::_CopyFileInternal(
    const oss_request_options_t* oss_options, aos_pool_t* pool,
    const aos_string_t& source_bucket, const aos_string_t& source_object,
    const aos_string_t& dest_bucket, const aos_string_t& dest_object) {
  aos_status_t* resp_status;
  aos_table_t* resp_headers;
  aos_table_t* headers = aos_table_make(pool, 0);
  aos_string_t upload_id;

  oss_list_upload_part_params_t* list_upload_part_params;
  oss_upload_part_copy_params_t* upload_part_copy_params =
      oss_create_upload_part_copy_params(pool);
  oss_list_part_content_t* part_content;
  aos_list_t complete_part_list;
  oss_complete_part_content_t* complete_content;
  aos_table_t* list_part_resp_headers = NULL;
  aos_table_t* complete_resp_headers = NULL;
  int max_ret = 1000;

  // get file size
  TF_FileStatistics stat;
  _StatInternal(pool, oss_options, std::string(source_bucket.data),
                std::string(source_object.data), &stat);
  uint64 file_size = stat.length;

  // file size bigger than upload_part_bytes_, need to split into multi parts
  if (file_size > upload_part_bytes_) {
    resp_status =
        oss_init_multipart_upload(oss_options, &dest_bucket, &dest_object,
                                  &upload_id, headers, &resp_headers);
    if (aos_status_is_ok(resp_status)) {
      VLOG(1) << "init multipart upload succeeded, upload_id is %s"
              << upload_id.data;
    } else {
      return resp_status;
    }

    // process for each single part
    int parts = ceil(double(file_size) / double(upload_part_bytes_));
    for (int i = 0; i < parts - 1; i++) {
      int64_t range_start = i * upload_part_bytes_;
      int64_t range_end = (i + 1) * upload_part_bytes_ - 1;
      int part_num = i + 1;

      aos_str_set(&upload_part_copy_params->source_bucket, source_bucket.data);
      aos_str_set(&upload_part_copy_params->source_object, source_object.data);
      aos_str_set(&upload_part_copy_params->dest_bucket, dest_bucket.data);
      aos_str_set(&upload_part_copy_params->dest_object, dest_object.data);
      aos_str_set(&upload_part_copy_params->upload_id, upload_id.data);

      upload_part_copy_params->part_num = part_num;
      upload_part_copy_params->range_start = range_start;
      upload_part_copy_params->range_end = range_end;

      headers = aos_table_make(pool, 0);

      resp_status = oss_upload_part_copy(oss_options, upload_part_copy_params,
                                         headers, &resp_headers);
      if (aos_status_is_ok(resp_status)) {
        VLOG(1) << "upload part " << part_num << " copy succeeded";
      } else {
        return resp_status;
      }
    }

    int64_t range_start = (parts - 1) * upload_part_bytes_;
    int64_t range_end = file_size - 1;

    aos_str_set(&upload_part_copy_params->source_bucket, source_bucket.data);
    aos_str_set(&upload_part_copy_params->source_object, source_object.data);
    aos_str_set(&upload_part_copy_params->dest_bucket, dest_bucket.data);
    aos_str_set(&upload_part_copy_params->dest_object, dest_object.data);
    aos_str_set(&upload_part_copy_params->upload_id, upload_id.data);
    upload_part_copy_params->part_num = parts;
    upload_part_copy_params->range_start = range_start;
    upload_part_copy_params->range_end = range_end;

    headers = aos_table_make(pool, 0);

    resp_status = oss_upload_part_copy(oss_options, upload_part_copy_params,
                                       headers, &resp_headers);
    if (aos_status_is_ok(resp_status)) {
      VLOG(1) << "upload part " << parts << " copy succeeded";
    } else {
      return resp_status;
    }

    headers = aos_table_make(pool, 0);
    list_upload_part_params = oss_create_list_upload_part_params(pool);
    list_upload_part_params->max_ret = max_ret;
    aos_list_init(&complete_part_list);
    resp_status = oss_list_upload_part(oss_options, &dest_bucket, &dest_object,
                                       &upload_id, list_upload_part_params,
                                       &list_part_resp_headers);
    aos_list_for_each_entry(oss_list_part_content_t, part_content,
                            &list_upload_part_params->part_list, node) {
      complete_content = oss_create_complete_part_content(pool);
      aos_str_set(&complete_content->part_number,
                  part_content->part_number.data);
      aos_str_set(&complete_content->etag, part_content->etag.data);
      aos_list_add_tail(&complete_content->node, &complete_part_list);
    }

    resp_status = oss_complete_multipart_upload(
        oss_options, &dest_bucket, &dest_object, &upload_id,
        &complete_part_list, headers, &complete_resp_headers);
    if (aos_status_is_ok(resp_status)) {
      VLOG(1) << "complete multipart upload succeeded";
    }
  } else {
    resp_status =
        oss_copy_object(oss_options, &source_bucket, &source_object,
                        &dest_bucket, &dest_object, headers, &resp_headers);
  }

  return resp_status;
}

Status OSSFileSystem::IsDirectory(const std::string& fname) {
  TF_FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(fname, &stat));

  return stat.is_directory
             ? OkStatus()
             : errors::FailedPrecondition(fname + " is not a directory");
}

Status OSSFileSystem::DeleteRecursively(const std::string& dirname,
                                        uint64* undeleted_files,
                                        uint64* undeleted_dirs) {
  if (!undeleted_files || !undeleted_dirs) {
    return errors::Internal(
        "'undeleted_files' and 'undeleted_dirs' cannot be nullptr.");
  }
  *undeleted_files = 0;
  *undeleted_dirs = 0;

  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dirname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  std::vector<std::string> children;

  TF_FileStatistics stat;
  Status s;
  s = _StatInternal(pool, oss_options, bucket, object, &stat);
  if (!s.ok() || !stat.is_directory) {
    *undeleted_dirs = 1;
    return errors::NotFound(dirname, " doesn't exist or not a directory.");
  }

  s = _ListObjects(pool, oss_options, bucket, object, &children, true, true,
                   false, true, 1000);
  if (!s.ok()) {
    // empty dir, just delete it
    return _DeleteObjectInternal(oss_options, bucket, object);
  }

  for (const auto& child : children) {
    s = _DeleteObjectInternal(oss_options, bucket, child);
    if (!s.ok()) {
      s = _StatInternal(pool, oss_options, bucket, child, &stat);
      if (s.ok()) {
        if (stat.is_directory) {
          ++*undeleted_dirs;
        } else {
          ++*undeleted_files;
        }
      }
    }
  }

  if (*undeleted_dirs == 0 && *undeleted_files == 0) {
    // delete directory itself.
    if (object.at(object.length() - 1) == '/') {
      return _DeleteObjectInternal(oss_options, bucket, object);
    } else {
      return _DeleteObjectInternal(oss_options, bucket, object.append(kDelim));
    }
  }
  return OkStatus();
}

Status OSSFileSystem::CopyFile(const string& src, const string& target) {
  TF_RETURN_IF_ERROR(oss_initialize());

  std::string sobject, sbucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(src, sbucket, sobject, host, access_id, access_key));
  std::string dobject, dbucket;
  std::string dhost, daccess_id, daccess_key;
  TF_RETURN_IF_ERROR(_ParseOSSURIPath(target, dbucket, dobject, dhost,
                                      daccess_id, daccess_key));

  if (host != dhost || access_id != daccess_id || access_key != daccess_key) {
    VLOG(0) << "rename " << src << " to " << target << " failed, with errMsg: "
            << " source oss cluster does not match dest oss cluster";
    return errors::Internal(
        "rename ", src, " to ", target, " failed, errMsg: ",
        "source oss cluster does not match dest oss cluster");
  }

  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();

  aos_status_t* resp_status;
  aos_string_t source_bucket;
  aos_string_t source_object;
  aos_string_t dest_bucket;
  aos_string_t dest_object;

  aos_str_set(&source_bucket, sbucket.c_str());
  aos_str_set(&source_object, sobject.c_str());
  aos_str_set(&dest_bucket, dbucket.c_str());
  aos_str_set(&dest_object, dobject.c_str());

  resp_status = _CopyFileInternal(oss_options, pool, source_bucket,
                                  source_object, dest_bucket, dest_object);
  if (!aos_status_is_ok(resp_status)) {
    string msg;
    oss_error_message(resp_status, &msg);
    VLOG(0) << "copy " << src << " to " << target << " failed, errMsg: " << msg;
    return errors::Internal("copy ", src, " to ", target,
                            " failed, errMsg: ", msg);
  }
  return OkStatus();
}

void ToTF_Status(const ::tensorflow::Status& s, TF_Status* status) {
  TF_SetStatus(status, TF_Code(int(s.code())), s.error_message().c_str());
}

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

static void Cleanup(TF_RandomAccessFile* file) {
  auto oss_file = static_cast<OSSRandomAccessFile*>(file->plugin_file);
  delete oss_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
  auto oss_file = static_cast<OSSRandomAccessFile*>(file->plugin_file);
  StringPiece result;
  ToTF_Status(oss_file->Read(offset, n, &result, buffer), status);
  return result.size();
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

static void Cleanup(TF_WritableFile* file) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  delete oss_file;
}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  ToTF_Status(oss_file->Append(StringPiece(buffer, n)), status);
}

static int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Stat not implemented");
  return -1;
}

static void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  ToTF_Status(oss_file->Flush(), status);
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  ToTF_Status(oss_file->Sync(), status);
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  ToTF_Status(oss_file->Close(), status);
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {
void Cleanup(TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<OSSReadOnlyMemoryRegion*>(region->plugin_memory_region);
  delete r;
}

const void* Data(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<OSSReadOnlyMemoryRegion*>(region->plugin_memory_region);
  return r->data();
}

uint64_t Length(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<OSSReadOnlyMemoryRegion*>(region->plugin_memory_region);
  return r->length();
}

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_oss_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  filesystem->plugin_filesystem = new OSSFileSystem();
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  delete oss_fs;
}

void NewRandomAccessFile(const TF_Filesystem* filesystem, const char* path,
                         TF_RandomAccessFile* file, TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  std::unique_ptr<RandomAccessFile> result;
  ToTF_Status(oss_fs->NewRandomAccessFile(path, &result), status);
  if (TF_GetCode(status) == TF_OK) {
    file->plugin_file = result.release();
  }
}

void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                     TF_WritableFile* file, TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  std::unique_ptr<WritableFile> result;
  ToTF_Status(oss_fs->NewWritableFile(path, &result), status);
  if (TF_GetCode(status) == TF_OK) {
    file->plugin_file = result.release();
  }
}

void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                       TF_WritableFile* file, TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  std::unique_ptr<WritableFile> result;
  ToTF_Status(oss_fs->NewAppendableFile(path, &result), status);
  if (TF_GetCode(status) == TF_OK) {
    file->plugin_file = result.release();
  }
}

void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                     const char* path,
                                     TF_ReadOnlyMemoryRegion* region,
                                     TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  std::unique_ptr<ReadOnlyMemoryRegion> result;
  ToTF_Status(oss_fs->NewReadOnlyMemoryRegionFromFile(path, &result), status);
  if (TF_GetCode(status) == TF_OK) {
    region->plugin_memory_region = result.release();
  }
}

void CreateDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  ToTF_Status(oss_fs->CreateDir(path), status);
}

void RecursivelyCreateDir(const TF_Filesystem* filesystem, const char* path,
                          TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  ToTF_Status(oss_fs->RecursivelyCreateDir(path), status);
}

void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  ToTF_Status(oss_fs->DeleteFile(path), status);
}

void DeleteDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  ToTF_Status(oss_fs->DeleteDir(path), status);
}

void DeleteRecursively(const TF_Filesystem* filesystem, const char* path,
                       uint64_t* undeleted_files, uint64_t* undeleted_dirs,
                       TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  ToTF_Status(oss_fs->DeleteRecursively(path, undeleted_files, undeleted_dirs),
              status);
}

void RenameFile(const TF_Filesystem* filesystem, const char* src,
                const char* dst, TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  ToTF_Status(oss_fs->RenameFile(src, dst), status);
}

void CopyFile(const TF_Filesystem* filesystem, const char* src, const char* dst,
              TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  ToTF_Status(oss_fs->CopyFile(src, dst), status);
}

bool IsDirectory(const TF_Filesystem* filesystem, const char* path,
                 TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  ToTF_Status(oss_fs->IsDirectory(path), status);
  return TF_GetCode(status) == TF_OK;
}

void Stat(const TF_Filesystem* filesystem, const char* path,
          TF_FileStatistics* stats, TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  ToTF_Status(oss_fs->Stat(path, stats), status);
}

void PathExists(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  TF_FileStatistics stats;
  Stat(filesystem, path, &stats, status);
}

int GetChildren(const TF_Filesystem* filesystem, const char* path,
                char*** entries, TF_Status* status) {
  auto oss_fs = static_cast<OSSFileSystem*>(filesystem->plugin_filesystem);
  std::vector<std::string> result;
  ToTF_Status(oss_fs->GetChildren(path, &result), status);
  int num_entries = result.size();
  *entries = static_cast<char**>(
      plugin_memory_allocate(num_entries * sizeof((*entries)[0])));
  for (int i = 0; i < num_entries; i++)
    (*entries)[i] = strdup(result[i].c_str());
  return TF_GetCode(status) == TF_OK ? num_entries : -1;
}

int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                    TF_Status* status) {
  TF_FileStatistics stats;
  Stat(filesystem, path, &stats, status);
  return stats.length;
}

char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  return strdup(uri);
}

}  // namespace tf_oss_filesystem

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
  ops->filesystem_ops->init = tf_oss_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_oss_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_oss_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_oss_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_oss_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_oss_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_oss_filesystem::CreateDir;
  ops->filesystem_ops->recursively_create_dir =
      tf_oss_filesystem::RecursivelyCreateDir;
  ops->filesystem_ops->delete_file = tf_oss_filesystem::DeleteFile;
  ops->filesystem_ops->delete_recursively =
      tf_oss_filesystem::DeleteRecursively;
  ops->filesystem_ops->delete_dir = tf_oss_filesystem::DeleteDir;
  ops->filesystem_ops->copy_file = tf_oss_filesystem::CopyFile;
  ops->filesystem_ops->rename_file = tf_oss_filesystem::RenameFile;
  ops->filesystem_ops->path_exists = tf_oss_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_oss_filesystem::Stat;
  ops->filesystem_ops->is_directory = tf_oss_filesystem::IsDirectory;
  ops->filesystem_ops->get_file_size = tf_oss_filesystem::GetFileSize;
  ops->filesystem_ops->get_children = tf_oss_filesystem::GetChildren;
  ops->filesystem_ops->translate_name = tf_oss_filesystem::TranslateName;
}

}  // end namespace oss
}  // end namespace io
}  // end namespace tensorflow
