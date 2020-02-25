/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_io/core/kernels/audio_kernels.h"

#define MINIMP3_IMPLEMENTATION
#include "minimp3_ex.h"

namespace tensorflow {
namespace data {
namespace {

class MP3Stream {
 public:
  MP3Stream(SizedRandomAccessFile* file, int64 size)
      : file(file), size(size), offset(0) {}
  ~MP3Stream() {}

  static size_t ReadCallback(void* buf, size_t size, void* user_data) {
    MP3Stream* p = static_cast<MP3Stream*>(user_data);
    StringPiece result;
    Status status = p->file->Read(p->offset, size, &result, (char*)buf);
    p->offset += result.size();
    return result.size();
  }

  static int SeekCallback(uint64_t position, void* user_data) {
    MP3Stream* p = static_cast<MP3Stream*>(user_data);
    if (position < 0 || position > p->size) {
      return -1;
    }
    p->offset = position;
    return 0;
  }

  SizedRandomAccessFile* file = nullptr;
  int64 size = 0;
  long offset = 0;
};

class MP3ReadableResource : public AudioReadableResourceBase {
 public:
  MP3ReadableResource(Env* env)
      : env_(env), mp3dec_ex_scope_(nullptr, [](mp3dec_ex_t* p) {
          if (p != nullptr) {
            mp3dec_ex_close(p);
          }
        }) {}
  ~MP3ReadableResource() {}

  Status Init(const string& input) override {
    mutex_lock l(mu_);
    const string& filename = input;
    file_.reset(new SizedRandomAccessFile(env_, filename, nullptr, 0));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    stream_.reset(new MP3Stream(file_.get(), file_size_));

    mp3dec_io_.read = MP3Stream::ReadCallback;
    mp3dec_io_.read_data = stream_.get();
    mp3dec_io_.seek = MP3Stream::SeekCallback;
    mp3dec_io_.seek_data = stream_.get();
    memset(&mp3dec_ex_, 0x00, sizeof(mp3dec_ex_));
    if (mp3dec_ex_open_cb(&mp3dec_ex_, &mp3dec_io_, MP3D_SEEK_TO_SAMPLE)) {
      return errors::InvalidArgument("unable to open file ", filename,
                                     " as mp3: ", mp3dec_ex_.last_error);
    }
    mp3dec_ex_scope_.reset(&mp3dec_ex_);
    if (mp3dec_ex_.info.channels == 0) {
      return errors::InvalidArgument("invalid mp3 with channel == 0");
    }
    int64 samples = mp3dec_ex_.samples / mp3dec_ex_.info.channels;
    int64 channels = mp3dec_ex_.info.channels;
    int64 rate = mp3dec_ex_.info.hz;

    shape_ = TensorShape({samples, channels});
    dtype_ = DT_INT16;
    rate_ = rate;

    return Status::OK();
  }

  Status Spec(TensorShape* shape, DataType* dtype, int32* rate) override {
    mutex_lock l(mu_);
    *shape = shape_;
    *dtype = dtype_;
    *rate = rate_;
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop,
              std::function<Status(const TensorShape& shape, Tensor** value)>
                  allocate_func) override {
    mutex_lock l(mu_);

    int64 sample_stop =
        (stop < 0) ? (shape_.dim_size(0))
                   : (stop < shape_.dim_size(0) ? stop : shape_.dim_size(0));
    int64 sample_start = (start >= sample_stop) ? sample_stop : start;

    Tensor* value;
    TF_RETURN_IF_ERROR(allocate_func(
        TensorShape({sample_stop - sample_start, shape_.dim_size(1)}), &value));

    if (mp3dec_ex_seek(&mp3dec_ex_, sample_start * shape_.dim_size(1))) {
      return errors::InvalidArgument("seek to ", sample_start,
                                     " failed: ", mp3dec_ex_.last_error);
    }
    size_t returned = mp3dec_ex_read(&mp3dec_ex_, value->flat<int16>().data(),
                                     value->NumElements());
    if (returned != value->NumElements()) {
      return errors::InvalidArgument("read ", value->NumElements(), " from ",
                                     sample_start,
                                     " failed: ", mp3dec_ex_.last_error);
    }
    return Status::OK();
  }
  string DebugString() const override { return "MP3ReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  std::unique_ptr<MP3Stream> stream_;
  mp3dec_io_t mp3dec_io_;
  mp3dec_ex_t mp3dec_ex_;
  std::unique_ptr<mp3dec_ex_t, void (*)(mp3dec_ex_t*)> mp3dec_ex_scope_;
};

}  // namespace

Status MP3ReadableResourceInit(
    Env* env, const string& input,
    std::unique_ptr<AudioReadableResourceBase>& resource) {
  resource.reset(new MP3ReadableResource(env));
  Status status = resource->Init(input);
  if (!status.ok()) {
    resource.reset(nullptr);
  }
  return status;
}

}  // namespace data
}  // namespace tensorflow
