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

#include <curl/curl.h>

#include <iostream>
#include <string>
#include <unordered_map>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_io/core/filesystems/filesystem_plugins.h"

namespace tensorflow {
namespace io {
namespace http {
namespace {

// Set to 1 to enable verbose debug output from curl.
constexpr uint64_t kVerboseOutput = 0;

static absl::Mutex mu;
static bool initialized(false);
void CurlInitialize() {
  absl::MutexLock l(&mu);
  if (!initialized) {
    curl_global_init(CURL_GLOBAL_ALL);
    initialized = true;
  }
}

class CurlHttpRequest {
 public:
  CurlHttpRequest() {}
  ~CurlHttpRequest() {}

  void Initialize(TF_Status* status) {
    CurlInitialize();
    curl_ = curl_easy_init();
    if (curl_ == nullptr) {
      TF_SetStatus(status, TF_INTERNAL, "Couldn't initialize a curl session.");
      return;
    }

    CURLcode s = CURLE_OK;

    const char* ca_bundle = std::getenv("CURL_CA_BUNDLE");
    if (ca_bundle != nullptr) {
      if ((s = curl_easy_setopt(curl_, CURLOPT_CAINFO, ca_bundle)) !=
          CURLE_OK) {
        std::string error_message =
            absl::StrCat("Unable to set CURLOPT_CAINFO (", ca_bundle, "): ", s);
        TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
        return;
      }
    }

    if ((s = curl_easy_setopt(curl_, CURLOPT_VERBOSE, kVerboseOutput)) !=
        CURLE_OK) {
      std::string error_message = absl::StrCat(
          "Unable to set CURLOPT_VERBOSE (", kVerboseOutput, "): ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    if ((s = curl_easy_setopt(curl_, CURLOPT_USERAGENT,
                              absl::StrCat("TensorFlowIO/", 0).c_str())) !=
        CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_USERAGENT (",
                       absl::StrCat("TensorFlowIO/", 0), "): ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    // Do not use signals for timeouts - does not work in multi-threaded
    // programs.
    if ((s = curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L)) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_NOSIGNAL: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    // TODO: Enable HTTP/2.
    if ((s = curl_easy_setopt(curl_, CURLOPT_HTTP_VERSION,
                              CURL_HTTP_VERSION_1_1)) != CURLE_OK) {
      std::string error_message = absl::StrCat(
          "Unable to set CURLOPT_HTTP_VERSION (CURL_HTTP_VERSION_1_1): ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    // Set up the progress meter.
    if ((s = curl_easy_setopt(curl_, CURLOPT_NOPROGRESS, 0)) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_NOPROGRESS (0): ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    if ((s = curl_easy_setopt(curl_, CURLOPT_XFERINFODATA, this)) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_XFERINFODATA: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    if ((s = curl_easy_setopt(curl_, CURLOPT_XFERINFOFUNCTION,
                              &CurlHttpRequest::ProgressCallback)) !=
        CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_XFERINFOFUNCTION: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    TF_SetStatus(status, TF_OK, "");
  }

  void SetUri(const std::string& uri, TF_Status* status) {
    CURLcode s = CURLE_OK;
    if ((s = curl_easy_setopt(curl_, CURLOPT_URL, uri.c_str())) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_URL (", uri, "): ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    uri_ = uri;

    TF_SetStatus(status, TF_OK, "");
  }

  void SetRange(uint64_t start, uint64_t end, TF_Status* status) {
    CURLcode s = CURLE_OK;
    if ((s = curl_easy_setopt(curl_, CURLOPT_RANGE,
                              absl::StrCat(start, "-", end).c_str())) !=
        CURLE_OK) {
      std::string error_message = absl::StrCat("Unable to set CURLOPT_RANGE (",
                                               start, "-", end, "): ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    TF_SetStatus(status, TF_OK, "");
  }

  void SetResultBuffer(TF_Status* status) {
    CURLcode s = CURLE_OK;
    response_buffer_.reserve(CURL_MAX_WRITE_SIZE);
    if ((s = curl_easy_setopt(curl_, CURLOPT_WRITEDATA,
                              reinterpret_cast<void*>(this))) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_WRITEDATA: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }
    if ((s = curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION,
                              &CurlHttpRequest::WriteCallback)) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_WRITEFUNCTION ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    TF_SetStatus(status, TF_OK, "");
  }

  void SetResultBufferDirect(char* buffer, size_t size, TF_Status* status) {
    CURLcode s = CURLE_OK;
    direct_response_ = DirectResponseState{buffer, size, 0, 0};
    if ((s = curl_easy_setopt(curl_, CURLOPT_WRITEDATA,
                              reinterpret_cast<void*>(this))) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_WRITEDATA: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }
    if ((s = curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION,
                              &CurlHttpRequest::WriteCallbackDirect)) !=
        CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_WRITEFUNCTION: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }
    TF_SetStatus(status, TF_OK, "");
  }

  size_t GetResultBufferDirectBytesTransferred() {
    return direct_response_.bytes_transferred_;
  }

  std::string GetResponseHeader(const std::string& name) {
    const auto& header = response_headers_.find(name);
    return header != response_headers_.end() ? header->second : "";
  }

  void Send(TF_Status* status) {
    CURLcode s = CURLE_OK;

    if (curl_headers_) {
      if ((s = curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, curl_headers_)) !=
          CURLE_OK) {
        std::string error_message =
            absl::StrCat("Unable to set CURLOPT_HTTPHEADER: ", s);
        TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
        return;
      }
    }
    if (resolve_list_) {
      if ((s = curl_easy_setopt(curl_, CURLOPT_RESOLVE, resolve_list_)) !=
          CURLE_OK) {
        std::string error_message =
            absl::StrCat("Unable to set CURLOPT_RESOLVE: ", s);
        TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
        return;
      }
    }
    if ((s = curl_easy_setopt(curl_, CURLOPT_HEADERDATA,
                              reinterpret_cast<void*>(this))) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_HEADERDATA: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }
    if ((s = curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION,
                              &CurlHttpRequest::HeaderCallback)) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_HEADERFUNCTION: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    if ((s = curl_easy_setopt(curl_, CURLOPT_TIMEOUT, request_timeout_secs_)) !=
        CURLE_OK) {
      std::string error_message = absl::StrCat(
          "Unable to set CURLOPT_TIMEOUT (", request_timeout_secs_, "): ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }
    if ((s = curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT,
                              connect_timeout_secs_)) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_CONNECTTIMEOUT (",
                       connect_timeout_secs_, "): ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    char error_buffer[CURL_ERROR_SIZE] = {0};
    if ((s = curl_easy_setopt(curl_, CURLOPT_ERRORBUFFER, error_buffer)) !=
        CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLOPT_ERRORBUFFER: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    if ((s = curl_easy_perform(curl_)) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to perform (", s, "): ", error_buffer);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    double written_size = 0;
    if ((s = curl_easy_getinfo(curl_, CURLINFO_SIZE_DOWNLOAD, &written_size)) !=
        CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLINFO_SIZE_DOWNLOAD: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    if ((s = curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE,
                               &response_code_)) != CURLE_OK) {
      std::string error_message =
          absl::StrCat("Unable to set CURLINFO_RESPONSE_CODE: ", s);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    auto get_error_message = [this]() -> std::string {
      std::string error_message =
          absl::StrCat("Error executing an HTTP request: HTTP response code ",
                       response_code_);
      absl::string_view body = GetResponse();
      if (!body.empty()) {
        return absl::StrCat(
            error_message, " with body '",
            body.substr(0, std::min(body.size(), response_to_error_limit_)),
            "'");
      }
      return error_message;
    };

    switch (response_code_) {
      // The group of response codes indicating that the request achieved
      // the expected goal.
      case 200:  // OK
      case 201:  // Created
      case 204:  // No Content
      case 206:  // Partial Content
        TF_SetStatus(status, TF_OK, "");
        break;

      case 416:  // Requested Range Not Satisfiable
        // The requested range had no overlap with the available range.
        // This doesn't indicate an error, but we should produce an empty
        // response body. (Not all servers do; GCS returns a short error message
        // body.)
        response_buffer_.clear();
        if (IsDirectResponse()) {
          direct_response_.bytes_transferred_ = 0;
        }
        TF_SetStatus(status, TF_OK, "");
        break;

      // INVALID_ARGUMENT indicates a problem with how the request is
      // constructed.
      case 400:  // Bad Request
      case 406:  // Not Acceptable
      case 411:  // Length Required
      case 414:  // URI Too Long
        TF_SetStatus(status, TF_INVALID_ARGUMENT, get_error_message().c_str());
        break;

      // PERMISSION_DENIED indicates an authentication or an authorization
      // issue.
      case 401:  // Unauthorized
      case 403:  // Forbidden
      case 407:  // Proxy Authorization Required
        TF_SetStatus(status, TF_PERMISSION_DENIED, get_error_message().c_str());
        break;

      // NOT_FOUND indicates that the requested resource does not exist.
      case 404:  // Not found
      case 410:  // Gone
        TF_SetStatus(status, TF_NOT_FOUND, get_error_message().c_str());
        break;

      // FAILED_PRECONDITION indicates that the request failed because some
      // of the underlying assumptions were not satisfied. The request
      // shouldn't be retried unless the external context has changed.
      case 302:  // Found
      case 303:  // See Other
      case 304:  // Not Modified
      case 307:  // Temporary Redirect
      case 412:  // Precondition Failed
      case 413:  // Payload Too Large
        TF_SetStatus(status, TF_FAILED_PRECONDITION,
                     get_error_message().c_str());
        break;

      // UNAVAILABLE indicates a problem that can go away if the request
      // is just retried without any modification. 308 return codes are intended
      // for write requests that can be retried. See the documentation and the
      // official library:
      // https://cloud.google.com/storage/docs/json_api/v1/how-tos/resumable-upload
      // https://github.com/google/apitools/blob/master/apitools/base/py/transfer.py
      case 308:  // Resume Incomplete
      case 409:  // Conflict
      case 429:  // Too Many Requests
      case 500:  // Internal Server Error
      case 502:  // Bad Gateway
      case 503:  // Service Unavailable
      default:   // All other HTTP response codes also should be retried.
        TF_SetStatus(status, TF_UNAVAILABLE, get_error_message().c_str());
        break;
    }
    if (TF_GetCode(status) != TF_OK) {
      response_buffer_.clear();
    }
  }

 private:
  std::vector<char> response_buffer_;

  struct DirectResponseState {
    char* buffer_;
    size_t buffer_size_;
    size_t bytes_transferred_;
    size_t bytes_received_;
  };
  DirectResponseState direct_response_ = {};

  CURL* curl_ = nullptr;
  curl_slist* curl_headers_ = nullptr;
  curl_slist* resolve_list_ = nullptr;

  std::unordered_map<std::string, std::string> response_headers_;
  uint64_t response_code_ = 0;

  std::string uri_;

  // The timestamp of the last activity related to the request execution, in
  // seconds since epoch.
  uint64_t last_progress_timestamp_ = 0;
  // The last progress in terms of bytes transmitted.
  curl_off_t last_progress_bytes_ = 0;

  // The maximum period of request inactivity.
  uint32_t inactivity_timeout_secs_ = 60;  // 1 minute

  // Timeout for the connection phase.
  uint32_t connect_timeout_secs_ = 120;  // 2 minutes

  // Timeout for the whole request. Set only to prevent hanging indefinitely.
  uint32_t request_timeout_secs_ = 3600;  // 1 hour

  // Limit the size of an http response that is copied into an error message.
  const size_t response_to_error_limit_ = 500;

  bool IsDirectResponse() const { return direct_response_.buffer_ != nullptr; }

  absl::string_view GetResponse() const {
    absl::string_view response;
    if (IsDirectResponse()) {
      response = absl::string_view(direct_response_.buffer_,
                                   direct_response_.bytes_transferred_);
    } else {
      response =
          absl::string_view(response_buffer_.data(), response_buffer_.size());
    }
    return response;
  }

  // Cancels the transmission if no progress has been made for too long.
  static int ProgressCallback(void* this_object, curl_off_t dltotal,
                              curl_off_t dlnow, curl_off_t ultotal,
                              curl_off_t ulnow) {
    auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
    const int64_t now = absl::ToUnixSeconds(absl::Now());
    const auto current_progress = dlnow + ulnow;
    if (that->last_progress_timestamp_ == 0 ||
        current_progress > that->last_progress_bytes_) {
      // This is the first time the callback is called or some progress
      // was made since the last tick.
      that->last_progress_timestamp_ = now;
      that->last_progress_bytes_ = current_progress;
      return 0;
    }

    if (now - that->last_progress_timestamp_ > that->inactivity_timeout_secs_) {
      double lookup_time = -1;
      const auto lookup_time_status = curl_easy_getinfo(
          that->curl_, CURLINFO_NAMELOOKUP_TIME, &lookup_time);

      double connect_time = -1;
      const auto connect_time_status =
          curl_easy_getinfo(that->curl_, CURLINFO_CONNECT_TIME, &connect_time);

      double pretransfer_time = -1;
      const auto pretransfer_time_status = curl_easy_getinfo(
          that->curl_, CURLINFO_PRETRANSFER_TIME, &pretransfer_time);

      double starttransfer_time = -1;
      const auto starttransfer_time_status = curl_easy_getinfo(
          that->curl_, CURLINFO_STARTTRANSFER_TIME, &starttransfer_time);

      std::string error_message = absl::StrCat(
          "The transmission  of request ", (int64_t)(this_object),
          " (URI: ", that->uri_, ") has been stuck at ", current_progress,
          " of ", dltotal + ultotal, " bytes for ",
          now - that->last_progress_timestamp_,
          " seconds and will be aborted. CURL timing information: ",
          "lookup time: ", lookup_time, " (",
          curl_easy_strerror(lookup_time_status),
          "), connect time: ", connect_time, " (",
          curl_easy_strerror(connect_time_status),
          "), pre-transfer time: ", pretransfer_time, " (",
          curl_easy_strerror(pretransfer_time_status),
          "), start-transfer time: ", starttransfer_time, " (",
          curl_easy_strerror(starttransfer_time_status), ")");
      TF_Log(TF_ERROR, error_message.c_str());
      return 1;  // Will abort the request.
    }
    // No progress was made since the last call, but we should wait a bit
    // longer.
    return 0;
  }

  static size_t WriteCallback(const void* ptr, size_t size, size_t nmemb,
                              void* this_object) {
    auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
    const size_t bytes_to_copy = size * nmemb;
    that->response_buffer_.insert(
        that->response_buffer_.end(), reinterpret_cast<const char*>(ptr),
        reinterpret_cast<const char*>(ptr) + bytes_to_copy);

    return bytes_to_copy;
  }

  static size_t WriteCallbackDirect(const void* ptr, size_t size, size_t nmemb,
                                    void* userdata) {
    auto that = reinterpret_cast<CurlHttpRequest*>(userdata);
    DirectResponseState* state = &that->direct_response_;

    size_t curl_bytes_received = size * nmemb;
    size_t user_buffer_bytes_available =
        state->buffer_size_ - state->bytes_transferred_;
    size_t bytes_to_copy =
        std::min<size_t>(curl_bytes_received, user_buffer_bytes_available);
    memcpy(&state->buffer_[state->bytes_transferred_], ptr, bytes_to_copy);
    state->bytes_transferred_ += bytes_to_copy;
    state->bytes_received_ += curl_bytes_received;
    // If we didn't have room to store the full response, returning less than
    // curl_bytes_received here will abort the transfer and curl_easy_perform()
    // will return CURLE_WRITE_ERROR. We will detect and handle this error
    // there, and can use state->bytes_received_ as stored above for logging
    // purposes.
    return bytes_to_copy;
  }
  static size_t HeaderCallback(const void* ptr, size_t size, size_t nmemb,
                               void* this_object) {
    auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
    absl::string_view header(reinterpret_cast<const char*>(ptr), size * nmemb);
    absl::string_view::size_type p = header.find(": ");
    if (p != absl::string_view::npos) {
      std::string name(header.substr(0, p));
      std::string value(header.substr(p + 2, -1));
      absl::StripTrailingAsciiWhitespace(&value);
      that->response_headers_[name] = value;
    }
    return size * nmemb;
  }
};

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
    CurlHttpRequest request;
    request.Initialize(status);
    if (TF_GetCode(status) != TF_OK) {
      return 0;
    }
    request.SetUri(uri_, status);
    if (TF_GetCode(status) != TF_OK) {
      return 0;
    }
    request.SetRange(offset, offset + n - 1, status);
    if (TF_GetCode(status) != TF_OK) {
      return 0;
    }
    request.SetResultBufferDirect(buffer, n, status);
    if (TF_GetCode(status) != TF_OK) {
      return 0;
    }
    request.Send(status);
    if (TF_GetCode(status) != TF_OK) {
      return 0;
    }
    size_t bytes_to_read = request.GetResultBufferDirectBytesTransferred();
    if (bytes_to_read < n) {
      TF_SetStatus(status, TF_OUT_OF_RANGE, "EOF reached");
      return bytes_to_read;
    }
    TF_SetStatus(status, TF_OK, "");
    return bytes_to_read;
  }

 private:
  std::string uri_;
};

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
namespace tf_http_filesystem {

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
  CurlHttpRequest request;
  request.Initialize(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  request.SetResultBuffer(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  request.SetUri(path, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  request.Send(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  std::string length_string = request.GetResponseHeader("Content-Length");
  if (length_string == "") {
    std::string error_message =
        absl::StrCat("unable to check the Content-Length of the url: ", path);
    TF_SetStatus(status, TF_INVALID_ARGUMENT, error_message.c_str());
    return;
  }
  int64_t length = 0;
  if (!absl::SimpleAtoi<int64_t>(length_string, &length)) {
    std::string error_message =
        absl::StrCat("unable to parse the Content-Length of the url: ", path,
                     " [", length_string, "]");
    TF_SetStatus(status, TF_INVALID_ARGUMENT, error_message.c_str());
    return;
  }

  std::string last_modified_string = request.GetResponseHeader("Last-Modified");

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

}  // namespace tf_http_filesystem

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
  ops->filesystem_ops->init = tf_http_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_http_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_http_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_http_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_http_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_http_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_http_filesystem::CreateDir;
  ops->filesystem_ops->recursively_create_dir =
      tf_http_filesystem::RecursivelyCreateDir;
  ops->filesystem_ops->delete_file = tf_http_filesystem::DeleteFile;
  ops->filesystem_ops->delete_recursively =
      tf_http_filesystem::DeleteRecursively;
  ops->filesystem_ops->delete_dir = tf_http_filesystem::DeleteDir;
  ops->filesystem_ops->copy_file = tf_http_filesystem::CopyFile;
  ops->filesystem_ops->rename_file = tf_http_filesystem::RenameFile;
  ops->filesystem_ops->path_exists = tf_http_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_http_filesystem::Stat;
  ops->filesystem_ops->is_directory = tf_http_filesystem::IsDirectory;
  ops->filesystem_ops->get_file_size = tf_http_filesystem::GetFileSize;
  ops->filesystem_ops->get_children = tf_http_filesystem::GetChildren;
  ops->filesystem_ops->translate_name = tf_http_filesystem::TranslateName;
}

}  // namespace http
}  // namespace io
}  // namespace tensorflow
