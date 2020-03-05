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

#include "vorbis/codec.h"
#include "vorbis/vorbisfile.h"

namespace tensorflow {
namespace data {
namespace {

class OggVorbisStream {
 public:
  OggVorbisStream(SizedRandomAccessFile* file, int64 size)
      : file(file), size(size), offset(0) {}
  ~OggVorbisStream() {}

  static size_t ReadCallback(void* ptr, size_t size, size_t count,
                             void* stream) {
    OggVorbisStream* p = static_cast<OggVorbisStream*>(stream);
    StringPiece result;
    Status status = p->file->Read(p->offset, size * count, &result, (char*)ptr);
    size_t items = result.size() / size;
    p->offset += items * size;
    return items;
  }

  static int SeekCallback(void* stream, ogg_int64_t offset, int origin) {
    OggVorbisStream* p = static_cast<OggVorbisStream*>(stream);
    switch (origin) {
      case SEEK_SET:
        if (offset < 0 || offset > p->size) {
          return -1;
        }
        p->offset = offset;
        break;
      case SEEK_CUR:
        if (p->offset + offset < 0 || p->offset + offset > p->size) {
          return -1;
        }
        p->offset += offset;
        break;
      case SEEK_END:
        if (p->size + offset < 0 || p->size + offset > p->size) {
          return -1;
        }
        p->offset = p->size + offset;
        break;
      default:
        return -1;
    }
    return 0;
  }

  static long TellCallback(void* stream) {
    OggVorbisStream* p = static_cast<OggVorbisStream*>(stream);
    return p->offset;
  }

  SizedRandomAccessFile* file = nullptr;
  int64 size = 0;
  long offset = 0;
};

static ov_callbacks OggVorbisCallbacks = {
    (size_t(*)(void*, size_t, size_t, void*))OggVorbisStream::ReadCallback,
    (int (*)(void*, ogg_int64_t, int))OggVorbisStream::SeekCallback,
    (int (*)(void*))NULL, (long (*)(void*))OggVorbisStream::TellCallback};

class OggReadableResource : public AudioReadableResourceBase {
 public:
  OggReadableResource(Env* env) : env_(env) {}
  ~OggReadableResource() {}

  Status Init(const string& input) override {
    mutex_lock l(mu_);
    const string& filename = input;
    file_.reset(new SizedRandomAccessFile(env_, filename, nullptr, 0));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    stream_.reset(new OggVorbisStream(file_.get(), file_size_));
    int returned = ov_open_callbacks(stream_.get(), &ogg_vorbis_file_, NULL, -1,
                                     OggVorbisCallbacks);
    if (returned < 0) {
      return errors::InvalidArgument(
          "could not open input as an OggVorbis file: ", returned);
    }

    vorbis_info* vi = ov_info(&ogg_vorbis_file_, -1);
    int64 samples = ov_pcm_total(&ogg_vorbis_file_, -1);
    int64 channels = vi->channels;
    int64 rate = vi->rate;

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

    int returned = ov_pcm_seek(&ogg_vorbis_file_, sample_start);
    if (returned < 0) {
      return errors::InvalidArgument("seek failed: ", returned);
    }

    int bitstream = 0;
    long bytes_read = 0;
    long bytes_to_read = value->NumElements() * sizeof(int16);
    while (bytes_read < bytes_to_read) {
      long chunk = ov_read(&ogg_vorbis_file_,
                           (char*)value->flat<int16>().data() + bytes_read,
                           bytes_to_read - bytes_read, 0, 2, 1, &bitstream);
      if (chunk < 0) {
        return errors::InvalidArgument("read failed: ", chunk);
      }
      if (chunk == 0) {
        return errors::InvalidArgument("not enough data: ");
      }
      bytes_read += chunk;
    }
    return Status::OK();
  }
  string DebugString() const override { return "OggReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  OggVorbis_File ogg_vorbis_file_;
  std::unique_ptr<OggVorbisStream> stream_;
};

}  // namespace

Status OggReadableResourceInit(
    Env* env, const string& input,
    std::unique_ptr<AudioReadableResourceBase>& resource) {
  resource.reset(new OggReadableResource(env));
  Status status = resource->Init(input);
  if (!status.ok()) {
    resource.reset(nullptr);
  }
  return status;
}

}  // namespace data
}  // namespace tensorflow
