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

#include "tensorflow/core/platform/file_system.h"
#include "tiff.h"
#include "tiffio.h"

namespace tensorflow {
namespace data {


// Base object for wrapping TIFFClientOpen file operation callbacks

class TiffFileBase {
  std::unique_ptr<TIFF, decltype(&TIFFClose)> tif_;

 public:
  TIFF* Tiff() { return tif_.get(); }

  TiffFileBase() :  tif_(nullptr, TIFFClose) {}
  virtual ~TiffFileBase() {}

 protected:
  Status ClientOpen(const char *name, const char* mode);


  void ClientClose() {tif_.reset(); }

 public:
  virtual size_t  TiffClientRead(void*, size_t) { return 0;}
  virtual size_t  TiffClientWrite(void*, size_t) { return 0; }
  virtual off_t   TiffClientSeek(off_t offset, int whence) = 0;
  virtual int     TiffClientClose() { return 0;}
  virtual off_t   TiffClientSize() = 0;
  virtual int     TiffClientMap(void** base, off_t* size) { return 0;}
  virtual void    TiffClientUnmap(void* base, off_t size) {}
};

// Use RandomAccessFile for tiff file system operation callbacks
class TiffRandomFile : public TiffFileBase {
 public:
  TiffRandomFile() {}
  ~TiffRandomFile() override  {
    Close();
  }

  Status Open(Env *env, const string& filename) {
    // Open Random access file
    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));
    // Read Size and hope we get same file as above
    uint64 size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &size));
    // Open Tiff
    fileSize_ = static_cast<off_t>(size);
    offset_ = 0;
    file_ = std::move(file);
    Status s = ClientOpen(filename.c_str(), "rm");
    if (!s.ok()) {
      file_.reset();
    }
    return s;
  }

  bool IsOpen() {
    return static_cast<bool>(file_);
  }

  void Close() {
    ClientClose();
    file_.reset();
  }

 private:
  size_t  TiffClientRead(void*, size_t) override;
  off_t   TiffClientSeek(off_t offset, int whence) override;
  int     TiffClientClose() override;
  off_t   TiffClientSize() override;

  std::unique_ptr<RandomAccessFile> file_;
  off_t fileSize_;
  off_t offset_;
};

}  // namespace data
}  // namespace tensorflow
