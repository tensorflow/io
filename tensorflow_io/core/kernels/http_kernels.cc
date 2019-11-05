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

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/cloud/curl_http_request.h"

namespace tensorflow {

class HTTPRandomAccessFile : public RandomAccessFile {
 public:
  HTTPRandomAccessFile(const std::string& uri, HttpRequest::Factory* http_request_factory)
   : uri_(uri)
   , http_request_factory_(http_request_factory) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    // If n == 0, then return Status::OK()
    // otherwise, if bytes_read < n then return OutofRange
    if (n == 0) {
      *result = StringPiece("", 0);
      return Status::OK();
    }
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    request->SetUri(uri_);
    request->SetRange(offset, offset + n - 1);
    request->SetResultBufferDirect(scratch, n);
    TF_RETURN_IF_ERROR(request->Send());
    size_t bytes_read = request->GetResultBufferDirectBytesTransferred();
    *result = StringPiece(scratch, bytes_read);
    if (bytes_read < n) {
      return errors::OutOfRange("EOF reached");
    }
    return Status::OK();
  }

 private:
  string uri_;
  HttpRequest::Factory* http_request_factory_;
};

class HTTPFileSystem : public FileSystem {
 public:
  HTTPFileSystem() {
    http_request_factory_ = std::make_shared<CurlHttpRequest::Factory>();
  }
  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override {
    result->reset(new HTTPRandomAccessFile(fname, http_request_factory_.get()));
    return Status::OK();
  };
  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewWritableFile");
  }
  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewAppendableFile");
  }
  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return errors::Unimplemented("NewReadOnlyMemoryRegionFromFile");
  }
  Status FileExists(const string& fname) override {
    return errors::Unimplemented("FileExists");
  }
  Status GetChildren(const string& dir,
                     std::vector<string>* result) override {
    return errors::Unimplemented("GetChildren");
  }
  Status GetMatchingPaths(const string& pattern,
                          std::vector<string>* results) override {
    return errors::Unimplemented("GetMatchingPaths");
  }
  Status Stat(const string& fname, FileStatistics* stat) override {
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    request->SetUri(fname);
    TF_RETURN_IF_ERROR(request->Send());
    string length_string = request->GetResponseHeader("Content-Length");
    if (length_string == "") {
      return errors::InvalidArgument("unable to check the Content-Length of the url: ", fname);
    }
    int64 length = 0;
    if (!strings::safe_strto64(length_string, &length)) {
      return errors::InvalidArgument("unable to parse the Content-Length of the url: ", fname, " [", length_string, "]");
    }

    string last_modified_string = request->GetResponseHeader("Last-Modified");

    FileStatistics fs;
    fs.length = length;
    fs.mtime_nsec = 0;
    *stat = std::move(fs);

    return Status::OK();
  }
  Status DeleteFile(const string& fname) override {
    return errors::Unimplemented("DeleteFile");
  }
  Status CreateDir(const string& dirname) override {
    return errors::Unimplemented("CreateDir");
  }
  Status DeleteDir(const string& dirname) override {
    return errors::Unimplemented("DeleteDir");
  }
  Status GetFileSize(const string& fname, uint64* file_size) override {
    FileStatistics stat;
    TF_RETURN_IF_ERROR(Stat(fname, &stat));
    *file_size = stat.length;
    return Status::OK();
  }
  Status RenameFile(const string& src, const string& target) override {
    return errors::Unimplemented("RenameFile");
  }
 private:
  mutex mu_;
  std::shared_ptr<HttpRequest::Factory> http_request_factory_;
};

REGISTER_FILE_SYSTEM("http", HTTPFileSystem);

}  // namespace tensorflow
