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

#define MINIMP4_IMPLEMENTATION
#include "minimp4.h"

extern "C" {
#if defined(__APPLE__)
void DecodeAACFunctionFini(void* state) { return; }
void* DecodeAACFunctionInit(const int64_t codec, const int64_t rate,
                            const int64_t channels) {
  return (void*)1;  // return 1 to pretend success
}
int64_t DecodeAACFunctionCall(void* state, const int64_t codec,
                              const int64_t rate, const int64_t channels,
                              const int64_t frames, const void* data_in,
                              int64_t size_in, void* data_out,
                              int64_t size_out);
#elif defined(_MSC_VER)
void DecodeAACFunctionFini(void* state) { return; }
void* DecodeAACFunctionInit(const int64_t codec, const int64_t rate,
                            const int64_t channels) {
  return nullptr;
}
int64_t DecodeAACFunctionCall(void* state, const int64_t codec,
                              const int64_t rate, const int64_t channels,
                              const int64_t frames, const void* data_in,
                              int64_t size_in, void* data_out,
                              int64_t size_out) {
  return -1;
}
#else
#include <dlfcn.h>
static void (*DecodeAACFunctionFiniPointer)(void* state);
static void* (*DecodeAACFunctionInitPointer)(const int64_t codec,
                                             const int64_t rate,
                                             const int64_t channels);
static int64_t (*DecodeAACFunctionCallPointer)(
    void* state, const int64_t codec, const int64_t rate,
    const int64_t channels, const int64_t frames, const void* data_in,
    int64_t size_in, void* data_out, int64_t size_out);

void DecodeAACFunctionFini(void* state) {
  if (DecodeAACFunctionFiniPointer != nullptr) {
    return DecodeAACFunctionFiniPointer(state);
  }
  return;
}
void* DecodeAACFunctionInit(const int64_t codec, const int64_t rate,
                            const int64_t channels) {
  *(void**)(&DecodeAACFunctionFiniPointer) =
      dlsym(RTLD_DEFAULT, "DecodeAACFunctionFiniFFmpeg");
  *(void**)(&DecodeAACFunctionInitPointer) =
      dlsym(RTLD_DEFAULT, "DecodeAACFunctionInitFFmpeg");
  *(void**)(&DecodeAACFunctionCallPointer) =
      dlsym(RTLD_DEFAULT, "DecodeAACFunctionCallFFmpeg");
  if (DecodeAACFunctionFiniPointer == nullptr ||
      DecodeAACFunctionInitPointer == nullptr ||
      DecodeAACFunctionCallPointer == nullptr) {
    DecodeAACFunctionFiniPointer = nullptr;
    DecodeAACFunctionInitPointer = nullptr;
    DecodeAACFunctionCallPointer = nullptr;

    return nullptr;
  }
  return DecodeAACFunctionInitPointer(codec, rate, channels);
}
int64_t DecodeAACFunctionCall(void* state, const int64_t codec,
                              const int64_t rate, const int64_t channels,
                              const int64_t frames, const void* data_in,
                              int64_t size_in, void* data_out,
                              int64_t size_out) {
  if (DecodeAACFunctionCallPointer != nullptr) {
    return DecodeAACFunctionCallPointer(state, codec, rate, channels, frames,
                                        data_in, size_in, data_out, size_out);
  }
  return -1;
}
#endif
}

namespace tensorflow {
namespace data {
namespace {

class MP4Stream {
 public:
  MP4Stream(SizedRandomAccessFile* file, int64 size) : file(file), size(size) {}
  ~MP4Stream() {}

  static int ReadCallback(int64_t offset, void* buffer, size_t size,
                          void* token) {
    MP4Stream* p = static_cast<MP4Stream*>(token);
    int64 bytes_to_read =
        (offset + size < p->size) ? (size) : (p->size - offset);
    StringPiece result;
    Status status =
        p->file->Read(offset, bytes_to_read, &result, (char*)buffer);
    return result.size() != bytes_to_read;
  }

  SizedRandomAccessFile* file = nullptr;
  int64 size = 0;
};

class MP4ReadableResource : public AudioReadableResourceBase {
 public:
  MP4ReadableResource(Env* env)
      : env_(env),
        mp4d_demux_scope_(nullptr,
                          [](MP4D_demux_t* p) {
                            if (p != nullptr) {
                              MP4D_close(p);
                            }
                          }),
        state_(nullptr, [](void* p) {
          if (p != nullptr) {
            DecodeAACFunctionFini(p);
          }
        }) {}
  ~MP4ReadableResource() {}

  Status Init(const string& input) override {
    mutex_lock l(mu_);
    const string& filename = input;
    file_.reset(new SizedRandomAccessFile(env_, filename, nullptr, 0));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    stream_.reset(new MP4Stream(file_.get(), file_size_));
    memset(&mp4d_demux_, 0x00, sizeof(mp4d_demux_));
    if (!MP4D_open(&mp4d_demux_, MP4Stream::ReadCallback, stream_.get(),
                   file_size_)) {
      return errors::InvalidArgument("unable to open file ", filename,
                                     " as mp4");
    }
    mp4d_demux_scope_.reset(&mp4d_demux_);

    for (int64 track_index = 0; track_index < mp4d_demux_.track_count;
         track_index++) {
      if (mp4d_demux_.track[track_index].handler_type ==
          MP4D_HANDLER_TYPE_SOUN) {
        int64 samples = 0;
        for (int64 i = 0; i < mp4d_demux_.track[track_index].sample_count;
             i++) {
          unsigned frame_bytes, timestamp, duration;
          MP4D_frame_offset(&mp4d_demux_, track_index, i, &frame_bytes,
                            &timestamp, &duration);
          samples += duration;
        }
        int64 channels =
            mp4d_demux_.track[track_index].SampleDescription.audio.channelcount;
        int64 rate = mp4d_demux_.track[track_index]
                         .SampleDescription.audio.samplerate_hz;

        int64 codec = 0;  // TODO
        state_.reset(DecodeAACFunctionInit(codec, rate, channels));
        if (state_.get() == nullptr) {
          return errors::InvalidArgument("unable to initialize mp4 state");
        }

        int64 profile = 2;  // AAC LC (Low Complexity)
        int64 channel_configuration =
            mp4d_demux_.track[track_index].SampleDescription.audio.channelcount;
        int64 frequency_index = -1;
        static const int64 frequency_indices[] = {
            96000, 88200, 64000, 48000, 44100, 32000, 24000,
            22050, 16000, 12000, 11025, 8000,  7350,
        };
        for (int64 i = 0;
             i < sizeof(frequency_indices) / sizeof(frequency_indices[0]);
             i++) {
          if (frequency_indices[i] ==
              mp4d_demux_.track[track_index]
                  .SampleDescription.audio.samplerate_hz) {
            frequency_index = i;
            break;
          }
        }
        if (frequency_index < 0) {
          return errors::InvalidArgument(
              "sample rate is not supported: ",
              mp4d_demux_.track[track_index]
                  .SampleDescription.audio.samplerate_hz);
        }

        track_index_ = track_index;
        codec_ = codec;
        profile_ = profile;
        channel_configuration_ = channel_configuration;
        frequency_index_ = frequency_index;

        shape_ = TensorShape({samples, channels});
        dtype_ = DT_FLOAT;
        rate_ = rate;
        return Status::OK();
      }
    }

    return errors::InvalidArgument("unable to find audio track for ", input);
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

    if (sample_stop == sample_start) {
      return Status::OK();
    }

    int64 sample_index = 0;
    for (int64 i = 0; i < mp4d_demux_.track[track_index_].sample_count; i++) {
      unsigned frame_bytes, timestamp, duration;
      MP4D_file_offset_t frame_offset = MP4D_frame_offset(
          &mp4d_demux_, track_index_, i, &frame_bytes, &timestamp, &duration);
      // [sample_start, sample_stop) in [sample_index, sample_index + duration)
      int64 sample_copy_start =
          sample_start > sample_index ? sample_start : sample_index;
      int64 sample_copy_stop = sample_stop < (sample_index + duration)
                                   ? sample_stop
                                   : (sample_index + duration);
      if (sample_copy_start < sample_copy_stop) {
        void* state = state_.get();
        int64 codec = codec_;
        int64 rate = rate_;
        int64 channels = shape_.dim_size(1);
        int64 frames = duration;

        int64 header_bytes = 7;
        int64 size_in = frame_bytes + header_bytes;
        int64 size_out =
            duration * channels * sizeof(float);  // TODO: expand to other types

        string data_in, data_out;
        data_in.resize(size_in);
        data_out.resize(size_out);

        StringPiece result;
        TF_RETURN_IF_ERROR(file_->Read(frame_offset, frame_bytes, &result,
                                       (char*)&data_in[header_bytes]));
        if (result.size() != frame_bytes) {
          return errors::InvalidArgument(
              "unable to read ", frame_bytes, " from offset ", frame_offset,
              " for track ", track_index_, " and sample indices in ", i);
        }

        // Add ADTS Header (without CRC)
        *((unsigned char*)&data_in[0]) = 0xFF;
        *((unsigned char*)&data_in[1]) = 0xF1;
        *((unsigned char*)&data_in[2]) =
            (((profile_ - 1) << 6) + (frequency_index_ << 2) +
             (channel_configuration_ >> 2));
        ;
        *((unsigned char*)&data_in[3]) =
            (((channel_configuration_ & 3) << 6) + (size_in >> 11));
        *((unsigned char*)&data_in[4]) = (((size_in & 0x07FF) >> 3));
        *((unsigned char*)&data_in[5]) = (((size_in & 0x0007) << 5) + 0x1F);
        *((unsigned char*)&data_in[6]) = 0xFC;

        int64 status = DecodeAACFunctionCall(
            state, codec, rate, channels, frames, (void*)&data_in[0], size_in,
            (void*)&data_out[0], size_out);
        if (status != 0) {
          return errors::InvalidArgument("unable to convert AAC data: ",
                                         status);
        }
        char* base =
            (char*)(value->flat<float>().data()) +
            (sample_copy_start - sample_start) * channels * sizeof(float);
        char* source =
            (char*)(&data_out[0]) +
            (sample_copy_start - sample_index) * channels * sizeof(float);
        size_t size =
            (sample_copy_stop - sample_copy_start) * channels * sizeof(float);
        memcpy(base, source, size);
      }
      sample_index += duration;
    }

    return Status::OK();
  }
  string DebugString() const override { return "MP4ReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  std::unique_ptr<MP4Stream> stream_;
  MP4D_demux_t mp4d_demux_;
  std::unique_ptr<MP4D_demux_t, void (*)(MP4D_demux_t*)> mp4d_demux_scope_;

  std::unique_ptr<void, void (*)(void*)> state_;

  int64 track_index_;

  int64 codec_;

  int64 profile_;
  int64 channel_configuration_;
  int64 frequency_index_;
};

}  // namespace

Status MP4ReadableResourceInit(
    Env* env, const string& input,
    std::unique_ptr<AudioReadableResourceBase>& resource) {
  resource.reset(new MP4ReadableResource(env));
  Status status = resource->Init(input);
  if (!status.ok()) {
    resource.reset(nullptr);
  }
  return status;
}

}  // namespace data
}  // namespace tensorflow
