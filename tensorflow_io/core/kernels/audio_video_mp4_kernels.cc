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
void EncodeAACFunctionFini(void* state);
void* EncodeAACFunctionInit(const int64_t codec, const int64_t rate,
                            const int64_t channels);
int64_t EncodeAACFunctionCall(void* state, const float* data_in,
                              const int64_t size_in, char** data_out_chunk,
                              int64_t* size_out_chunk, int64_t* chunk);

void DecodeAACFunctionFini(void* state) { return; }
void* DecodeAACFunctionInit(const int64_t codec, const int64_t rate,
                            const int64_t channels) {
  return (void*)1;  // return 1 to pretend success
}
int64_t DecodeAACFunctionCall(void* state, const int64_t codec,
                              const int64_t rate, const int64_t channels,
                              const void* data_in_chunk,
                              const int64_t* size_in_chunk, int64_t chunk,
                              int64_t frames, void* data_out, int64_t size_out);

void DecodeAVCFunctionFini(void* context);
void* DecodeAVCFunctionInit(const uint8_t* data_pps, const int64_t size_pps,
                            const uint8_t* data_sps, const int64_t size_sps,
                            int64_t* width, int64_t* height, int64_t* bytes);
int64_t DecodeAVCFunctionNext(void* context, const void* data_in,
                              int64_t size_in, void* data_out,
                              int64_t size_out);

#elif defined(_MSC_VER)
void EncodeAACFunctionFini(void* state) { return; }
void* EncodeAACFunctionInit(const int64_t codec, const int64_t rate,
                            const int64_t channels) {
  return nullptr;
}
int64_t EncodeAACFunctionCall(void* state, const float* data_in,
                              const int64_t size_in, char** data_out_chunk,
                              int64_t* size_out_chunk, int64_t* chunk) {
  return -1;
}

void DecodeAACFunctionFini(void* state) { return; }
void* DecodeAACFunctionInit(const int64_t codec, const int64_t rate,
                            const int64_t channels) {
  return nullptr;
}
int64_t DecodeAACFunctionCall(void* state, const int64_t codec,
                              const int64_t rate, const int64_t channels,
                              const void* data_in_chunk,
                              const int64_t* size_in_chunk, int64_t chunk,
                              int64_t frames, void* data_out,
                              int64_t size_out) {
  return -1;
}

void DecodeAVCFunctionFini(void* context) { return; }
void* DecodeAVCFunctionInit(const uint8_t* data_pps, const int64_t size_pps,
                            const uint8_t* data_sps, const int64_t size_sps,
                            int64_t* width, int64_t* height, int64_t* bytes) {
  return nullptr;
}
int64_t DecodeAVCFunctionNext(void* context, const void* data_in,
                              int64_t size_in, void* data_out,
                              int64_t size_out) {
  return -1;
}

#else
#include <dlfcn.h>
static void (*EncodeAACFunctionFiniPointer)(void* state);
static void* (*EncodeAACFunctionInitPointer)(const int64_t codec,
                                             const int64_t rate,
                                             const int64_t channels);
static int64_t (*EncodeAACFunctionCallPointer)(
    void* state, const float* data_in, const int64_t size_in,
    char** data_out_chunk, int64_t* size_out_chunk, int64_t* chunk);

void* EncodeAACFunctionInit(const int64_t codec, const int64_t rate,
                            const int64_t channels) {
  *(void**)(&EncodeAACFunctionFiniPointer) =
      dlsym(RTLD_DEFAULT, "EncodeAACFunctionFiniFFmpeg");
  *(void**)(&EncodeAACFunctionInitPointer) =
      dlsym(RTLD_DEFAULT, "EncodeAACFunctionInitFFmpeg");
  *(void**)(&EncodeAACFunctionCallPointer) =
      dlsym(RTLD_DEFAULT, "EncodeAACFunctionCallFFmpeg");
  if (EncodeAACFunctionFiniPointer == nullptr ||
      EncodeAACFunctionInitPointer == nullptr ||
      EncodeAACFunctionCallPointer == nullptr) {
    EncodeAACFunctionFiniPointer = nullptr;
    EncodeAACFunctionInitPointer = nullptr;
    EncodeAACFunctionCallPointer = nullptr;
    return nullptr;
  }
  return EncodeAACFunctionInitPointer(codec, rate, channels);
}
void EncodeAACFunctionFini(void* state) {
  if (EncodeAACFunctionFiniPointer != nullptr) {
    return EncodeAACFunctionFiniPointer(state);
  }
  return;
}
int64_t EncodeAACFunctionCall(void* state, const float* data_in,
                              const int64_t size_in, char** data_out_chunk,
                              int64_t* size_out_chunk, int64_t* chunk) {
  if (EncodeAACFunctionCallPointer != nullptr) {
    return EncodeAACFunctionCallPointer(state, data_in, size_in, data_out_chunk,
                                        size_out_chunk, chunk);
  }
  return -1;
}

static void (*DecodeAACFunctionFiniPointer)(void* state);
static void* (*DecodeAACFunctionInitPointer)(const int64_t codec,
                                             const int64_t rate,
                                             const int64_t channels);
static int64_t (*DecodeAACFunctionCallPointer)(
    void* state, const int64_t codec, const int64_t rate,
    const int64_t channels, const void* data_in_chunk,
    const int64_t* size_in_chunk, int64_t chunk, int64_t frames, void* data_out,
    int64_t size_out);

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
                              const void* data_in_chunk,
                              const int64_t* size_in_chunk, int64_t chunk,
                              int64_t frames, void* data_out,
                              int64_t size_out) {
  if (DecodeAACFunctionCallPointer != nullptr) {
    return DecodeAACFunctionCallPointer(state, codec, rate, channels,
                                        data_in_chunk, size_in_chunk, chunk,
                                        frames, data_out, size_out);
  }
  return -1;
}

void DecodeAVCFunctionFini(void* context) { return; }
void* DecodeAVCFunctionInit(const uint8_t* data_pps, const int64_t size_pps,
                            const uint8_t* data_sps, const int64_t size_sps,
                            int64_t* width, int64_t* height, int64_t* bytes) {
  return nullptr;
}
int64_t DecodeAVCFunctionNext(void* context, const void* data_in,
                              int64_t size_in, void* data_out,
                              int64_t size_out) {
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

class MP4AACReadableResource : public AudioReadableResourceBase {
 public:
  MP4AACReadableResource(Env* env)
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
  ~MP4AACReadableResource() {}

  Status Init(const string& filename, const void* optional_memory,
              const size_t optional_length) override {
    mutex_lock l(mu_);
    file_.reset(new SizedRandomAccessFile(env_, filename, optional_memory,
                                          optional_length));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    stream_.reset(new MP4Stream(file_.get(), file_size_));

    // reset the scope as resource might be reused
    mp4d_demux_scope_.reset(nullptr);
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
        if (mp4d_demux_.track[track_index].sample_count < preroll_ + padding_) {
          return errors::InvalidArgument(
              "need at least ", preroll_ + padding_,
              " packets: ", mp4d_demux_.track[track_index].sample_count);
        }

        int indication = mp4d_demux_.track[track_index].object_type_indication;
        int64 samples = 0;
        partitions_.clear();
        partitions_.reserve(mp4d_demux_.track[track_index].sample_count);

        for (int64 i = preroll_;
             i < mp4d_demux_.track[track_index].sample_count; i++) {
          unsigned frame_bytes, timestamp, duration;
          MP4D_frame_offset(&mp4d_demux_, track_index, i, &frame_bytes,
                            &timestamp, &duration);
          samples += duration;
          partitions_.emplace_back(samples);
        }
        samples = partitions_[(partitions_.size() - 1) - padding_];
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

    return errors::InvalidArgument("unable to find audio track for ", filename);
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

    void* state = state_.get();
    int64 codec = codec_;
    int64 rate = rate_;
    int64 channels = shape_.dim_size(1);

    int64 lower, upper, extra;
    TF_RETURN_IF_ERROR(PartitionsLookup(partitions_, sample_start, sample_stop,
                                        &lower, &upper, &extra));

    // we need append padding_ at the end.
    upper += padding_;

    static const int64 header_bytes = 7;

    int64 bytes = 0;
    int64 frames = 0;
    for (int64 i = lower; i < upper; i++) {
      unsigned frame_bytes, timestamp, duration;
      MP4D_file_offset_t frame_offset = MP4D_frame_offset(
          &mp4d_demux_, track_index_, i, &frame_bytes, &timestamp, &duration);

      bytes += frame_bytes + header_bytes;
      frames += duration;
    }
    string data_in_chunk;
    data_in_chunk.resize(bytes);
    std::vector<int64> size_in_chunk;

    int64 offset = 0;
    for (int64 i = lower; i < upper; i++) {
      unsigned frame_bytes, timestamp, duration;
      MP4D_file_offset_t frame_offset = MP4D_frame_offset(
          &mp4d_demux_, track_index_, i, &frame_bytes, &timestamp, &duration);

      char* data_in = (char*)&data_in_chunk[offset];
      int64 size_in = frame_bytes + header_bytes;

      StringPiece result;
      TF_RETURN_IF_ERROR(file_->Read(frame_offset, frame_bytes, &result,
                                     (char*)&data_in[header_bytes]));
      if (result.size() != frame_bytes) {
        return errors::InvalidArgument(
            "unable to read ", frame_bytes, " from offset ", frame_offset,
            " for track ", track_index_, " and sample indices in ", i);
      }
      size_in_chunk.push_back(size_in);

      // Add ADTS Header (without CRC)
      *((unsigned char*)&data_in[0]) = 0xFF;
      *((unsigned char*)&data_in[1]) = 0xF1;
      *((unsigned char*)&data_in[2]) =
          (((profile_ - 1) << 6) + (frequency_index_ << 2) +
           (channel_configuration_ >> 2));
      *((unsigned char*)&data_in[3]) =
          (((channel_configuration_ & 3) << 6) + (size_in >> 11));
      *((unsigned char*)&data_in[4]) = (((size_in & 0x07FF) >> 3));
      *((unsigned char*)&data_in[5]) = (((size_in & 0x0007) << 5) + 0x1F);
      *((unsigned char*)&data_in[6]) = 0xFC;

      offset += size_in;
    }

    int64 size_out = frames * channels * sizeof(float);
    string data_out;
    data_out.resize(size_out);
    int64 status =
        DecodeAACFunctionCall(state, codec, rate, channels, &data_in_chunk[0],
                              (int64_t*)&size_in_chunk[0], size_in_chunk.size(),
                              frames, (void*)&data_out[0], size_out);
    if (status != 0) {
      return errors::InvalidArgument("unable to convert AAC data: ", status);
    }
    char* base = (char*)(value->flat<float>().data());
    char* data = (char*)&data_out[0] + extra * channels * sizeof(float);
    memcpy(base, data, value->NumElements() * sizeof(float));
    return Status::OK();
  }
  string DebugString() const override { return "MP4AACReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
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

  std::vector<int64> partitions_;

  // decoder delay for preroll, and padding at the end?
  const int64 preroll_ = 0;
  const int64 padding_ = 1;
};

class AudioDecodeAACOp : public OpKernel {
 public:
  explicit AudioDecodeAACOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->input("shape", &shape_tensor));

    const tstring& input = input_tensor->scalar<tstring>()();

    std::unique_ptr<MP4AACReadableResource> resource(
        new MP4AACReadableResource(env_));
    OP_REQUIRES_OK(context,
                   resource->Init("memory", input.data(), input.size()));

    int32 rate;
    DataType dtype;
    TensorShape shape;
    OP_REQUIRES_OK(context, resource->Spec(&shape, &dtype, &rate));

    OP_REQUIRES(context, (dtype == context->expected_output_dtype(0)),
                errors::InvalidArgument(
                    "dtype mismatch: ", DataTypeString(dtype), " vs. ",
                    DataTypeString(context->expected_output_dtype(0))));

    PartialTensorShape provided_shape;
    OP_REQUIRES_OK(context, PartialTensorShape::MakePartialShape(
                                shape_tensor->flat<int64>().data(),
                                shape_tensor->NumElements(), &provided_shape));
    OP_REQUIRES(context, (provided_shape.IsCompatibleWith(shape)),
                errors::InvalidArgument(
                    "shape mismatch: ", provided_shape.DebugString(), " vs. ",
                    shape.DebugString()));

    OP_REQUIRES_OK(
        context,
        resource->Read(0, shape.dim_size(0),
                       [&](const TensorShape& shape, Tensor** value) -> Status {
                         TF_RETURN_IF_ERROR(
                             context->allocate_output(0, shape, value));
                         return Status::OK();
                       }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

static int AudioEncodeMP4AACWriteCallback(int64_t offset, const void* buffer,
                                          size_t size, void* token) {
  tstring* p = static_cast<tstring*>(token);
  if (offset + size > p->size()) {
    p->resize(offset + size);
  }
  if (size > 0) {
    memcpy(&(*p)[offset], buffer, size);
  }
  return 0;
}

class AudioEncodeAACOp : public OpKernel {
 public:
  explicit AudioEncodeAACOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* rate_tensor;
    OP_REQUIRES_OK(context, context->input("rate", &rate_tensor));

    const int64 channels = input_tensor->shape().dim_size(1);
    OP_REQUIRES(
        context, (channels == static_cast<int16>(channels)),
        errors::InvalidArgument("channels ", channels, " > max(int16)"));

    const int64 rate = rate_tensor->scalar<int64>()();
    OP_REQUIRES(context, (rate == static_cast<int32>(rate)),
                errors::InvalidArgument("rate ", rate, " > max(int32)"));

    // Code from buildAacAudioSpecificConfig:
    // ExoPlayer/library/core/src/main/java/com/google/android/exoplayer2/util/CodecSpecificDataUtil.java
    static const int AUDIO_SPECIFIC_CONFIG_SAMPLING_RATE_TABLE[] = {
        96000, 88200, 64000, 48000, 44100, 32000, 24000,
        22050, 16000, 12000, 11025, 8000,  7350};
    static const int AUDIO_SPECIFIC_CONFIG_CHANNEL_COUNT_TABLE[] = {
        0,  //
        1,  // mono: <FC>
        2,  // stereo: (FL, FR)
        3,  // 3.0: <FC>, (FL, FR)
        4,  // 4.0: <FC>, (FL, FR), <BC>
        5,  // 5.0 back: <FC>, (FL, FR), (SL, SR)
        6,  // 5.1 back: <FC>, (FL, FR), (SL, SR), <BC>, [LFE]
        8,  // 7.1 wide back: <FC>, (FCL, FCR), (FL, FR), (SL, SR), [LFE]
            // AUDIO_SPECIFIC_CONFIG_CHANNEL_CONFIGURATION_INVALID,
            // AUDIO_SPECIFIC_CONFIG_CHANNEL_CONFIGURATION_INVALID,
            // AUDIO_SPECIFIC_CONFIG_CHANNEL_CONFIGURATION_INVALID,
            // 7, // 6.1: <FC>, (FL, FR), (SL, SR), <RC>, [LFE]
            // 8, // 7.1: <FC>, (FL, FR), (SL, SR), (BL, BR), [LFE]
            // AUDIO_SPECIFIC_CONFIG_CHANNEL_CONFIGURATION_INVALID,
            // 8, // 7.1 top: <FC>, (FL, FR), (SL, SR), [LFE], (FTL, FTR)
            // AUDIO_SPECIFIC_CONFIG_CHANNEL_CONFIGURATION_INVALID
    };

    int audioObjectType = 2;  // AUDIO_OBJECT_TYPE_AAC_LC
    int sampleRateIndex = -1;
    for (int i = 0;
         i < sizeof(AUDIO_SPECIFIC_CONFIG_SAMPLING_RATE_TABLE) /
                 sizeof(AUDIO_SPECIFIC_CONFIG_SAMPLING_RATE_TABLE[0]);
         i++) {
      if (rate == AUDIO_SPECIFIC_CONFIG_SAMPLING_RATE_TABLE[i]) {
        sampleRateIndex = i;
        break;
      }
    }
    OP_REQUIRES(
        context, (sampleRateIndex >= 0),
        errors::InvalidArgument("sample rate ", rate, " not supported"));
    int channelConfig = -1;
    for (int i = 0;
         i < sizeof(AUDIO_SPECIFIC_CONFIG_CHANNEL_COUNT_TABLE) /
                 sizeof(AUDIO_SPECIFIC_CONFIG_CHANNEL_COUNT_TABLE[0]);
         i++) {
      if (channels == AUDIO_SPECIFIC_CONFIG_CHANNEL_COUNT_TABLE[i]) {
        channelConfig = i;
        break;
      }
    }
    OP_REQUIRES(
        context, (channelConfig >= 0),
        errors::InvalidArgument("channels ", channels, " not supported"));
    std::unique_ptr<void, void (*)(void*)> state(nullptr, [](void* p) {
      if (p != nullptr) {
        EncodeAACFunctionFini(p);
      }
    });

    state.reset(EncodeAACFunctionInit(0, rate, channels));
    OP_REQUIRES(context, (state.get() != nullptr),
                errors::InvalidArgument("unable to initialize encoder"));

    const float* data_in = input_tensor->flat<float>().data();
    const int64_t size_in = input_tensor->NumElements();

    int64_t chunk = input_tensor->NumElements() / channels / 1024 + 1;
    std::vector<char*> data_out_chunk((size_t)chunk);
    std::vector<int64_t> size_out_chunk((size_t)chunk);
    int status =
        EncodeAACFunctionCall(state.get(), data_in, size_in, &data_out_chunk[0],
                              &size_out_chunk[0], &chunk);
    OP_REQUIRES(context, (status == 0),
                errors::InvalidArgument("unable to encode aac"));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));

    tstring& output = output_tensor->scalar<tstring>()();
    int64 size_out = 0;
    for (int64 i = 0; i < chunk; i++) {
      size_out += size_out_chunk[i];
    }
    // At least size_out + 4096
    output.reserve(size_out + 4096);

    std::unique_ptr<MP4E_mux_t, void (*)(MP4E_mux_t*)> mux(nullptr,
                                                           [](MP4E_mux_t* p) {
                                                             if (p != nullptr) {
                                                               MP4E_close(p);
                                                               ;
                                                             }
                                                           });
    mux.reset(MP4E_open(0, 0, &output, AudioEncodeMP4AACWriteCallback));
    OP_REQUIRES(context, (mux.get() != nullptr),
                errors::InvalidArgument("unable open mux"));

    MP4E_track_t tr;
    tr.track_media_kind = e_audio;
    tr.language[0] = 'u';
    tr.language[1] = 'n';
    tr.language[2] = 'd';
    tr.language[3] = 0;
    tr.object_type_indication = MP4_OBJECT_TYPE_AUDIO_ISO_IEC_14496_3;
    tr.time_scale = rate;
    tr.default_duration = 0;
    tr.u.a.channelcount = channels;
    int audio_track_id = MP4E_add_track(mux.get(), &tr);

    unsigned char dsi[2];
    dsi[0] = (unsigned char)(((audioObjectType << 3) & 0xF8) |
                             ((sampleRateIndex >> 1) & 0x07));
    dsi[1] = (unsigned char)(((sampleRateIndex << 7) & 0x80) |
                             ((channelConfig << 3) & 0x78));

    status = MP4E_set_dsi(mux.get(), audio_track_id, dsi, sizeof(dsi));
    OP_REQUIRES(context, (status == 0),
                errors::InvalidArgument("unable to set dsi: ", status));

    for (int64 i = 0; i < chunk; i++) {
      status =
          MP4E_put_sample(mux.get(), audio_track_id, data_out_chunk[i],
                          size_out_chunk[i], 1024, MP4E_SAMPLE_RANDOM_ACCESS);
      OP_REQUIRES(
          context, (status == 0),
          errors::InvalidArgument("unable to mux packet ", i, ":", status));
    }
    // close mux
    mux.reset(nullptr);
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>AudioDecodeAAC").Device(DEVICE_CPU),
                        AudioDecodeAACOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioEncodeAAC").Device(DEVICE_CPU),
                        AudioEncodeAACOp);

}  // namespace

Status MP4AACReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource) {
  resource.reset(new MP4AACReadableResource(env));
  Status status = resource->Init(filename, optional_memory, optional_length);
  if (!status.ok()) {
    resource.reset(nullptr);
  }
  return status;
}

class VideoReadableResource : public ResourceBase {
 public:
  VideoReadableResource(Env* env)
      : env_(env),
        mp4d_demux_scope_(nullptr,
                          [](MP4D_demux_t* p) {
                            if (p != nullptr) {
                              MP4D_close(p);
                            }
                          }),
        context_(nullptr, [](void* p) {
          if (p != nullptr) {
            DecodeAACFunctionFini(p);
          }
        }) {}

  ~VideoReadableResource() {}

  Status Init(const string& filename) {
    mutex_lock l(mu_);

    file_.reset(new SizedRandomAccessFile(env_, filename, nullptr, 0));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    stream_.reset(new MP4Stream(file_.get(), file_size_));

    // reset the scope as resource might be reused
    mp4d_demux_scope_.reset(nullptr);
    memset(&mp4d_demux_, 0x00, sizeof(mp4d_demux_));
    if (!MP4D_open(&mp4d_demux_, MP4Stream::ReadCallback, stream_.get(),
                   file_size_)) {
      return errors::InvalidArgument("unable to open file ", filename,
                                     " as mp4");
    }
    mp4d_demux_scope_.reset(&mp4d_demux_);
    track_index_ = 0;
    sample_index_ = 0;
    while (track_index_ < mp4d_demux_.track_count) {
      if (mp4d_demux_.track[track_index_].handler_type ==
          MP4D_HANDLER_TYPE_VIDE) {
        break;
      }
    }
    if (track_index_ >= mp4d_demux_.track_count) {
      return errors::InvalidArgument("unable to find video stream from ",
                                     filename);
    }

    int size_pps = 0;
    const uint8_t* data_pps =
        (const uint8_t*)MP4D_read_pps(&mp4d_demux_, track_index_, 0, &size_pps);
    int size_sps = 0;
    const uint8_t* data_sps =
        (const uint8_t*)MP4D_read_sps(&mp4d_demux_, track_index_, 0, &size_sps);
    int64_t width, height, bytes;
    context_.reset(DecodeAVCFunctionInit(data_pps, size_pps, data_sps, size_sps,
                                         &width, &height, &bytes));
    if (context_.get() == nullptr) {
      return errors::InvalidArgument("unable to initialize mp4 state");
    }
    width_ = width;
    height_ = height;
    bytes_ = bytes;
    return Status::OK();
  }
  Status Read(
      const int64 index,
      std::function<Status(const TensorShape& shape, Tensor** value_tensor)>
          allocate_func) {
    mutex_lock l(mu_);

    if (index == 0) {
      sample_index_ = 0;
    }
    Tensor* value_tensor;
    if (sample_index_ >= mp4d_demux_.track[track_index_].sample_count) {
      TF_RETURN_IF_ERROR(allocate_func(TensorShape({0}), &value_tensor));
      return Status::OK();
    }

    unsigned frame_bytes, timestamp, duration;
    int64 off = MP4D_frame_offset(&mp4d_demux_, track_index_, sample_index_,
                                  &frame_bytes, &timestamp, &duration);
    string buffer;
    buffer.resize(frame_bytes);
    StringPiece result;
    TF_RETURN_IF_ERROR(
        file_->Read(off, frame_bytes, &result, (char*)&buffer[0]));
    if (result.size() != frame_bytes) {
      return errors::InvalidArgument("unable to read expected data of ",
                                     frame_bytes, " bytes at ", off);
    }

    TF_RETURN_IF_ERROR(allocate_func(TensorShape({1}), &value_tensor));
    tstring& value = value_tensor->flat<tstring>()(0);
    value.resize(bytes_);

    int64 status = DecodeAVCFunctionNext(
        context_.get(), (const void*)&buffer[0], frame_bytes, (void*)&value[0],
        static_cast<int64_t>(bytes_));
    if (status < 0) {
      return errors::InvalidArgument("error to decode: ", status);
    }
    sample_index_++;
    return Status::OK();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return "VideoReadableResource";
  }

 protected:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);

  std::unique_ptr<MP4Stream> stream_;
  MP4D_demux_t mp4d_demux_;
  std::unique_ptr<MP4D_demux_t, void (*)(MP4D_demux_t*)> mp4d_demux_scope_;
  std::unique_ptr<void, void (*)(void*)> context_;

  int64 width_;
  int64 height_;
  int64 bytes_;

  int64 track_index_;
  int64 sample_index_;
};

class VideoReadableInitOp : public ResourceOpKernel<VideoReadableResource> {
 public:
  explicit VideoReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<VideoReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<VideoReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(context, resource_->Init(input));
  }
  Status CreateResource(VideoReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new VideoReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class VideoReadableReadOp : public OpKernel {
 public:
  explicit VideoReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    VideoReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));
    const int64 index = index_tensor->scalar<int64>()();

    OP_REQUIRES_OK(context,
                   resource->Read(index,
                                  [&](const TensorShape& shape,
                                      Tensor** value_tensor) -> Status {
                                    TF_RETURN_IF_ERROR(context->allocate_output(
                                        0, shape, value_tensor));
                                    return Status::OK();
                                  }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>VideoReadableInit").Device(DEVICE_CPU),
                        VideoReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>VideoReadableRead").Device(DEVICE_CPU),
                        VideoReadableReadOp);
}  // namespace data
}  // namespace tensorflow
