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

#include <deque>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/io_stream.h"

extern "C" {

#include <dlfcn.h>

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
}

namespace tensorflow {

namespace {

bool initialized(false);
mutex mu(LINKER_INITIALIZED);

}  // namespace

namespace data {

void FFmpegInit() {
  mutex_lock lock(mu);
  if (!initialized) {
    // Set log level if needed
    static const struct {
      const char* name;
      int level;
    } log_levels[] = {
        {"quiet", AV_LOG_QUIET},     {"panic", AV_LOG_PANIC},
        {"fatal", AV_LOG_FATAL},     {"error", AV_LOG_ERROR},
        {"warning", AV_LOG_WARNING}, {"info", AV_LOG_INFO},
        {"verbose", AV_LOG_VERBOSE}, {"debug", AV_LOG_DEBUG},
        // { "trace"  , AV_LOG_TRACE   },
    };
    const char* log_level_name = getenv("FFMPEG_LOG_LEVEL");
    if (log_level_name != nullptr) {
      string log_level = log_level_name;
      for (size_t i = 0; i < sizeof(log_levels) / sizeof(log_levels[0]); i++) {
        if (log_level == log_levels[i].name) {
          LOG(INFO) << "FFmpeg log level: " << log_level;
          av_log_set_level(log_levels[i].level);
          break;
        }
      }
    }

    // Register all formats and codecs
    av_register_all();
    initialized = true;
  }
}

namespace {

class FFmpegStream {
 public:
  FFmpegStream(const string& filename, SizedRandomAccessFile* file,
               uint64 file_size)
      : filename_(filename),
        file_(file),
        file_size_(file_size),
        offset_(0),
        format_context_(nullptr,
                        [](AVFormatContext* p) {
                          if (p != nullptr) {
                            avformat_close_input(&p);
                          }
                        }),
        io_context_(nullptr,
                    [](AVIOContext* p) {
                      if (p != nullptr) {
                        av_free(p->buffer);
#if LIBAVCODEC_VERSION_MAJOR > 56
                        avio_context_free(&p);
#else
                        av_free(p);
#endif
                      }
                    }),
        stream_index_(-1),
        codec_context_(nullptr),
        codec_context_scope_(nullptr,
                             [](AVCodecContext* p) {
                               if (p != nullptr) {
                                 avcodec_free_context(&p);
                               }
                             }),
        nb_frames_(-1),
        packet_scope_(nullptr, [](AVPacket* p) {
          if (p != nullptr) {
            av_packet_unref(p);
          }
        }) {
  }
  virtual ~FFmpegStream() {}
  virtual Status Open(int64 media, int64 index) {
    constexpr int kIOBufferSize = 4096;  // TODO: maybe make this a parameter.
    int ret = 0;
    char error_message[AV_ERROR_MAX_STRING_SIZE];
    offset_ = 0;
    AVFormatContext* format_context = avformat_alloc_context();
    if (!format_context) {
      return errors::ResourceExhausted(
          "unable to allocate ffmpeg format context");
    }
    std::unique_ptr<AVFormatContext, decltype(avformat_free_context)*>
        format_context_scope(format_context, avformat_free_context);
    std::unique_ptr<uint8_t, decltype(av_free)*> io_buffer(
        (uint8_t*)av_malloc(kIOBufferSize), av_free);
    if (!io_buffer) {
      return errors::ResourceExhausted("unable to allocate ffmpeg io buffer");
    }
    io_context_.reset(avio_alloc_context(io_buffer.release(), kIOBufferSize, 0,
                                         this, FFmpegStream::ReadPacket, NULL,
                                         FFmpegStream::Seek));
    if (!io_context_) {
      return errors::ResourceExhausted("unable to allocate ffmpeg io context");
    }
    format_context->pb = io_context_.get();
    // Release here because avformat_open_input frees on failure.
    format_context_scope.release();
    ret = avformat_open_input(&format_context, filename_.c_str(), NULL, NULL);
    if (ret < 0) {
      av_strerror(ret, error_message, sizeof(error_message));
      return errors::InvalidArgument("unable to open file: ", filename_, ": ",
                                     error_message);
    }
    format_context_.reset(format_context);
    ret = avformat_find_stream_info(format_context_.get(), NULL);
    if (ret < 0) {
      av_strerror(ret, error_message, sizeof(error_message));
      return errors::InvalidArgument("unable to find stream info: ",
                                     error_message);
    }

    stream_index_ = -1;
    // No plan to read any other stream frame
    int64 media_index = 0;
    for (int64 i = 0; i < format_context->nb_streams; i++) {
#if LIBAVCODEC_VERSION_MAJOR > 56
      int media_type = format_context->streams[i]->codecpar->codec_type;
#else
      int media_type = format_context->streams[i]->codec->codec_type;
#endif
      if (media_type == media) {
        if (media_index == index) {
          stream_index_ = i;
        }
        media_index++;
      }
      if (stream_index_ != i) {
        format_context->streams[i]->discard = AVDISCARD_ALL;
      }
    }
    if (stream_index_ < 0) {
      return errors::InvalidArgument(
          "unable to find specified stream: media=", media, ", index=", index);
    }
    return Status::OK();
  }
  Status OpenCodec() {
    int64 stream_index = stream_index_;

#if LIBAVCODEC_VERSION_MAJOR > 56
    int codec_id = format_context_->streams[stream_index]->codecpar->codec_id;
#else
    int codec_id = format_context_->streams[stream_index]->codec->codec_id;
#endif
    // Find decoder for the stream
    AVCodec* codec = avcodec_find_decoder((enum AVCodecID)codec_id);
    if (codec == NULL) {
      return errors::InvalidArgument("unable to find codec id: ", codec_id);
    }
    codec_ = codec->name;
#if LIBAVCODEC_VERSION_MAJOR > 56
    codec_context_ = avcodec_alloc_context3(codec);
    if (codec_context_ == nullptr) {
      return errors::InvalidArgument("unable to allocate codec context");
    }
    codec_context_scope_.reset(codec_context_);
    // Copy codec parameters from input stream to output codec context
    if (avcodec_parameters_to_context(
            codec_context_, format_context_->streams[stream_index]->codecpar) <
        0) {
      return errors::Internal(
          "could not copy codec parameters from input stream to output codec "
          "context");
    }
#else
    codec_context_ = format_context_->streams[stream_index]->codec;
#endif
    {
      // avcodec_open2 is not thread-safe
      mutex_lock lock(mu);
      AVDictionary* opts = NULL;
      if (avcodec_open2(codec_context_, codec, &opts) < 0) {
        return errors::Internal("could not open codec");
      }
    }

    nb_frames_ = format_context_->streams[stream_index]->nb_frames;

    return Status::OK();
  }

  static int ReadPacket(void* opaque, uint8_t* buf, int buf_size) {
    FFmpegStream* r = (FFmpegStream*)opaque;
    StringPiece result;
    Status status = r->file_->Read(r->offset_, buf_size, &result, (char*)buf);
    if (!(status.ok() || errors::IsOutOfRange(status))) {
      return -1;
    }
    r->offset_ += result.size();
#if LIBAVFORMAT_VERSION_MAJOR > 57
    if (result.size() == 0) {
      return AVERROR_EOF;
    }
#endif
    return result.size();
  }

  static int64_t Seek(void* opaque, int64_t offset, int whence) {
    FFmpegStream* r = (FFmpegStream*)opaque;
    switch (whence) {
      case SEEK_SET:
        if (offset > r->file_size_) {
          return -1;
        }
        r->offset_ = offset;
        return r->offset_;
      case SEEK_CUR:
        if (r->offset_ + offset > r->file_size_) {
          return -1;
        }
        r->offset_ += offset;
        return r->offset_;
      case SEEK_END:
        if (offset > r->file_size_) {
          return -1;
        }
        r->offset_ = r->file_size_ - offset;
        return r->offset_;
      case AVSEEK_SIZE:
        return r->file_size_;
      default:
        break;
    }
    return -1;
  }

 public:
  string filename_;
  SizedRandomAccessFile* file_ = nullptr;
  uint64 file_size_ = 0;
  uint64 offset_ = 0;
  std::unique_ptr<AVFormatContext, void (*)(AVFormatContext*)> format_context_;
  std::unique_ptr<AVIOContext, void (*)(AVIOContext*)> io_context_;
  int64 stream_index_;
  string codec_;
  AVCodecContext* codec_context_;
  std::unique_ptr<AVCodecContext, void (*)(AVCodecContext*)>
      codec_context_scope_;
  int64 nb_frames_;
  AVPacket packet_;
  std::unique_ptr<AVPacket, void (*)(AVPacket*)> packet_scope_;
  std::deque<std::unique_ptr<AVFrame, void (*)(AVFrame*)>> frames_;
};

class FFmpegAudioStream : public FFmpegStream {
 public:
  FFmpegAudioStream(const string& filename, SizedRandomAccessFile* file,
                    uint64 file_size)
      : FFmpegStream(filename, file, file_size),
        dtype_(DT_INVALID),
        channels_(-1),
        rate_(-1) {}
  virtual ~FFmpegAudioStream() {}

  Status OpenAudio(int64 index) {
    TF_RETURN_IF_ERROR(Open(AVMEDIA_TYPE_AUDIO, index));
    TF_RETURN_IF_ERROR(OpenCodec());

    int64 stream_index = stream_index_;
#if LIBAVCODEC_VERSION_MAJOR > 56
    int format = format_context_->streams[stream_index]->codecpar->format;
    channels_ = format_context_->streams[stream_index]->codecpar->channels;
    rate_ = format_context_->streams[stream_index]->codecpar->sample_rate;
#else
    int format = format_context_->streams[stream_index]->codec->sample_fmt;
    channels_ = format_context_->streams[stream_index]->codec->channels;
    rate_ = format_context_->streams[stream_index]->codec->sample_rate;
#endif
    switch (format) {
      case AV_SAMPLE_FMT_U8:  ///< unsigned 8 bits
        dtype_ = DT_UINT8;
        break;
      case AV_SAMPLE_FMT_S16:  ///< signed 16 bits
        dtype_ = DT_INT16;
        break;
      case AV_SAMPLE_FMT_S32:  ///< signed 32 bits
        dtype_ = DT_INT32;
        break;
      case AV_SAMPLE_FMT_FLT:  ///< float
        dtype_ = DT_FLOAT;
        break;
      case AV_SAMPLE_FMT_DBL:  ///< double
        dtype_ = DT_DOUBLE;
        break;

      case AV_SAMPLE_FMT_U8P:   ///< unsigned 8 bits, planar
      case AV_SAMPLE_FMT_S16P:  ///< signed 16 bits, planar
      case AV_SAMPLE_FMT_S32P:  ///< signed 32 bits, planar
      case AV_SAMPLE_FMT_FLTP:  ///< float, planar
      case AV_SAMPLE_FMT_DBLP:  ///< double, planar
      // case AV_SAMPLE_FMT_S64:         ///< signed 64 bits
      // case AV_SAMPLE_FMT_S64P:        ///< signed 64 bits, planar
      default:
        return errors::InvalidArgument("invalid audio (", index,
                                       ") format: ", format);
    }
    int64 datasize = av_get_bytes_per_sample(codec_context_->sample_fmt);
    if (datasize != DataTypeSize(dtype_)) {
      return errors::InvalidArgument("failed to calculate data size");
    }
    // Initialize the decoders
    // Read first packet if possible
    av_init_packet(&packet_);
    packet_.data = NULL;
    packet_.size = 0;

    int ret = av_read_frame(format_context_.get(), &packet_);

    // reference after first
    packet_scope_.reset(&packet_);
    while (packet_.stream_index != stream_index_) {
      av_packet_unref(&packet_);
      ret = av_read_frame(format_context_.get(), &packet_);
      if (ret < 0) {
        av_packet_unref(&packet_);
        return errors::InvalidArgument("no frame available");
      }
    }
    int got_frame;
    while (packet_.size > 0) {
      TF_RETURN_IF_ERROR(DecodeFrame(&got_frame));
    }
    av_packet_unref(&packet_);

    return Status::OK();
  }
  Status Peek(int64* samples) {
    *samples = 0;
    TF_RETURN_IF_ERROR(DecodePacket());
    for (size_t i = 0; i < frames_.size(); i++) {
      (*samples) += frames_[i]->nb_samples;
    }
    return Status::OK();
  }
  Status Read(Tensor* value) {
    int64 datasize = DataTypeSize(dtype_);

    char* base;
    switch (dtype_) {
      case DT_INT16:
        base = ((char*)(value->flat<int16>().data()));
        break;
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       DataTypeString(dtype_));
    }
    // Note: only one channel supported so far
    for (size_t i = 0; i < frames_.size(); i++) {
      memcpy(base, (char*)(frames_[i]->extended_data[0]),
             datasize * frames_[i]->nb_samples);
      base += datasize * frames_[i]->nb_samples;
    }
    frames_.clear();
    return Status::OK();
  }

  Status DecodePacket() {
    if (packet_scope_.get() == nullptr) {
      return errors::OutOfRange("EOF reached");
    }
    int ret;
    do {
      av_packet_unref(&packet_);
      ret = av_read_frame(format_context_.get(), &packet_);
      if (ret < 0) {
        break;
      }
    } while (packet_.stream_index != stream_index_);
    int got_frame;
    // decode
    if (ret >= 0) {
      while (packet_.size > 0) {
        TF_RETURN_IF_ERROR(DecodeFrame(&got_frame));
      }
      av_packet_unref(&packet_);
      return Status::OK();
    }
    // final cache clean up
    do {
      TF_RETURN_IF_ERROR(DecodeFrame(&got_frame));
    } while (got_frame);
    packet_scope_.reset(nullptr);

    return Status::OK();
  }
  Status DecodeFrame(int* got_frame) {
    std::unique_ptr<AVFrame, void (*)(AVFrame*)> frame(av_frame_alloc(),
                                                       [](AVFrame* p) {
                                                         if (p != nullptr) {
                                                           av_frame_free(&p);
                                                         }
                                                       });
    int decoded =
        avcodec_decode_audio4(codec_context_, frame.get(), got_frame, &packet_);
    if (decoded < 0) {
      return errors::InvalidArgument("error decoding audio frame (", decoded,
                                     ")");
    }
    decoded = FFMIN(decoded, packet_.size);
    packet_.data += decoded;
    packet_.size -= decoded;
    if (*got_frame) {
      frames_.push_back(std::move(frame));
    }
    return Status::OK();
  }
  DataType dtype() { return dtype_; }
  int64 channels() { return channels_; }
  int64 rate() { return rate_; }

 private:
  DataType dtype_;
  int64 channels_;
  int64 rate_;
};

class FFmpegAudioReadableResource : public ResourceBase {
 public:
  FFmpegAudioReadableResource(Env* env) : env_(env) {}
  virtual ~FFmpegAudioReadableResource() {}

  Status Init(const string& input, const int64 index) {
    filename_ = input;
    audio_index_ = index;
    file_.reset(new SizedRandomAccessFile(env_, filename_, nullptr, 0));
    TF_RETURN_IF_ERROR(env_->GetFileSize(filename_, &file_size_));
    FFmpegInit();

    ffmpeg_audio_stream_.reset(
        new FFmpegAudioStream(filename_, file_.get(), file_size_));

    TF_RETURN_IF_ERROR(ffmpeg_audio_stream_->OpenAudio(audio_index_));

    sample_index_ = 0;

    return Status::OK();
  }
  Status Seek(const int64 index) {
    if (index != 0) {
      return errors::InvalidArgument("seek only support 0");
    }
    ffmpeg_audio_stream_.reset(
        new FFmpegAudioStream(filename_, file_.get(), file_size_));

    TF_RETURN_IF_ERROR(ffmpeg_audio_stream_->OpenAudio(audio_index_));
    sample_index_ = 0;
    return Status::OK();
  }
  Status Peek(TensorShape* shape) {
    int64 samples = 0;
    Status status = ffmpeg_audio_stream_->Peek(&samples);
    *shape = TensorShape({samples, ffmpeg_audio_stream_->channels()});
    return Status::OK();
  }
  Status Read(Tensor* value) { return ffmpeg_audio_stream_->Read(value); }
  string DebugString() const override { return "FFmpegAudioReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  string filename_ TF_GUARDED_BY(mu_);
  int64 audio_index_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
  std::unique_ptr<FFmpegAudioStream> ffmpeg_audio_stream_ TF_GUARDED_BY(mu_);
  int64 sample_index_ TF_GUARDED_BY(mu_);
};

class FFmpegAudioReadableInitOp
    : public ResourceOpKernel<FFmpegAudioReadableResource> {
 public:
  explicit FFmpegAudioReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<FFmpegAudioReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<FFmpegAudioReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));

    OP_REQUIRES_OK(context, resource_->Init(input_tensor->scalar<tstring>()(),
                                            index_tensor->scalar<int64>()()));
  }
  Status CreateResource(FFmpegAudioReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new FFmpegAudioReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class FFmpegAudioReadableNextOp : public OpKernel {
 public:
  explicit FFmpegAudioReadableNextOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    FFmpegAudioReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* reset_tensor;
    OP_REQUIRES_OK(context, context->input("reset", &reset_tensor));
    bool reset = reset_tensor->scalar<bool>()();
    if (reset) {
      OP_REQUIRES_OK(context, resource->Seek(0));
    }

    TensorShape value_shape;
    OP_REQUIRES_OK(context, resource->Peek(&value_shape));
    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, value_shape, &value_tensor));
    if (value_shape.dim_size(0) > 0) {
      OP_REQUIRES_OK(context, resource->Read(value_tensor));
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class FFmpegVideoStream : public FFmpegStream {
 public:
  FFmpegVideoStream(const string& filename, SizedRandomAccessFile* file,
                    uint64 file_size)
      : FFmpegStream(filename, file, file_size),
        dtype_(DT_INVALID),
        height_(-1),
        width_(-1),
        channels_(-1),
        sws_context_(nullptr, [](SwsContext* p) {
          if (p != nullptr) {
            sws_freeContext(p);
          }
        }) {}
  virtual ~FFmpegVideoStream() {}

  Status OpenVideo(int64 index) {
    TF_RETURN_IF_ERROR(Open(AVMEDIA_TYPE_VIDEO, index));
    TF_RETURN_IF_ERROR(OpenCodec());

    dtype_ = DT_UINT8;
    height_ = codec_context_->height;
    width_ = codec_context_->width;
    channels_ = 3;

    int64 datasize = av_image_get_buffer_size(
        AV_PIX_FMT_RGB24, codec_context_->width, codec_context_->height, 1);
    if (datasize != height_ * width_ * channels_) {
      return errors::InvalidArgument("failed to calculate data size");
    }

    SwsContext* sws_context = sws_getContext(
        codec_context_->width, codec_context_->height, codec_context_->pix_fmt,
        codec_context_->width, codec_context_->height, AV_PIX_FMT_RGB24, 0,
        NULL, NULL, NULL);
    if (!sws_context) {
      return errors::Internal("could not allocate sws context");
    }
    sws_context_.reset(sws_context);

    // Initialize the decoders
    // Read first packet if possible
    av_init_packet(&packet_);
    packet_.data = NULL;
    packet_.size = 0;

    int ret = av_read_frame(format_context_.get(), &packet_);

    // reference after first
    packet_scope_.reset(&packet_);
    while (packet_.stream_index != stream_index_) {
      av_packet_unref(&packet_);
      ret = av_read_frame(format_context_.get(), &packet_);
      if (ret < 0) {
        av_packet_unref(&packet_);
        return errors::InvalidArgument("no frame available");
      }
    }
    int got_frame;
    while (packet_.size > 0) {
      TF_RETURN_IF_ERROR(DecodeFrame(&got_frame));
    }
    av_packet_unref(&packet_);

    return Status::OK();
  }
  Status Peek(int64* frames) {
    *frames = 0;
    while (*frames == 0) {
      TF_RETURN_IF_ERROR(DecodePacket());
      (*frames) = frames_.size();
    }
    return Status::OK();
  }
  Status PeekAll(int64* frames) {
    Status status;
    do {
      status = DecodePacket();
    } while (status.ok());
    (*frames) = frames_.size();
    return Status::OK();
  }
  Status Read(Tensor* value) {
    char* base = ((char*)(value->flat<uint8>().data()));
    int64 datasize = height_ * width_ * channels_;
    for (size_t i = 0; i < frames_.size(); i++) {
      memcpy(base, reinterpret_cast<char*>(frames_buffer_.front().get()),
             datasize);
      base += datasize;
    }
    frames_.clear();
    frames_buffer_.clear();
    return Status::OK();
  }

  Status DecodePacket() {
    if (packet_scope_.get() == nullptr) {
      return errors::OutOfRange("EOF reached");
    }
    int ret;
    do {
      av_packet_unref(&packet_);
      ret = av_read_frame(format_context_.get(), &packet_);
      if (ret < 0) {
        break;
      }
    } while (packet_.stream_index != stream_index_);
    int got_frame;
    // decode
    if (ret >= 0) {
      while (packet_.size > 0) {
        TF_RETURN_IF_ERROR(DecodeFrame(&got_frame));
      }
      av_packet_unref(&packet_);
      return Status::OK();
    }
    // final cache clean up
    do {
      TF_RETURN_IF_ERROR(DecodeFrame(&got_frame));
    } while (got_frame);
    packet_scope_.reset(nullptr);

    return Status::OK();
  }
  Status DecodeFrame(int* got_frame) {
    std::unique_ptr<AVFrame, void (*)(AVFrame*)> frame(av_frame_alloc(),
                                                       [](AVFrame* p) {
                                                         if (p != nullptr) {
                                                           av_frame_free(&p);
                                                         }
                                                       });
    int decoded =
        avcodec_decode_video2(codec_context_, frame.get(), got_frame, &packet_);
    if (decoded < 0) {
      return errors::InvalidArgument("error decoding video frame (", decoded,
                                     ")");
    }
    decoded = FFMIN(decoded, packet_.size);
    packet_.data += decoded;
    packet_.size -= decoded;
    if (*got_frame) {
      int64 datasize = height_ * width_ * channels_;

      std::unique_ptr<AVFrame, void (*)(AVFrame*)> frame_rgb(
          av_frame_alloc(), [](AVFrame* p) {
            if (p != nullptr) {
              av_frame_free(&p);
            }
          });
      std::unique_ptr<uint8_t, void (*)(uint8_t*)> buffer_rgb(
          (uint8_t*)av_malloc(datasize), [](uint8_t* p) {
            if (p != nullptr) {
              av_free(p);
            }
          });
      avpicture_fill((AVPicture*)frame_rgb.get(), buffer_rgb.get(),
                     AV_PIX_FMT_RGB24, codec_context_->width,
                     codec_context_->height);
      sws_scale(sws_context_.get(), frame->data, frame->linesize, 0,
                codec_context_->height, frame_rgb->data, frame_rgb->linesize);

      frames_.push_back(std::move(frame_rgb));
      frames_buffer_.push_back(std::move(buffer_rgb));
    }
    return Status::OK();
  }
  DataType dtype() { return dtype_; }
  int64 channels() { return channels_; }
  int64 height() { return height_; }
  int64 width() { return width_; }

 private:
  DataType dtype_;

  int64 channels_;
  int64 height_;
  int64 width_;
  std::deque<std::unique_ptr<uint8_t, void (*)(uint8_t*)>> frames_buffer_;
  std::unique_ptr<SwsContext, void (*)(SwsContext*)> sws_context_;
};

class FFmpegVideoReadableResource : public ResourceBase {
 public:
  FFmpegVideoReadableResource(Env* env) : env_(env) {}
  virtual ~FFmpegVideoReadableResource() {}

  Status Init(const string& input, const int64 index) {
    filename_ = input;
    video_index_ = index;
    file_.reset(new SizedRandomAccessFile(env_, filename_, nullptr, 0));
    TF_RETURN_IF_ERROR(env_->GetFileSize(filename_, &file_size_));
    FFmpegInit();

    ffmpeg_video_stream_.reset(
        new FFmpegVideoStream(filename_, file_.get(), file_size_));

    TF_RETURN_IF_ERROR(ffmpeg_video_stream_->OpenVideo(video_index_));

    frame_index_ = 0;

    return Status::OK();
  }
  Status Seek(const int64 index) {
    if (index != 0) {
      return errors::InvalidArgument("seek only support 0");
    }
    ffmpeg_video_stream_.reset(
        new FFmpegVideoStream(filename_, file_.get(), file_size_));

    TF_RETURN_IF_ERROR(ffmpeg_video_stream_->OpenVideo(video_index_));
    frame_index_ = 0;
    return Status::OK();
  }
  Status Peek(TensorShape* shape) {
    int64 frames = 0;
    Status status = ffmpeg_video_stream_->Peek(&frames);
    *shape = TensorShape({frames, ffmpeg_video_stream_->height(),
                          ffmpeg_video_stream_->width(),
                          ffmpeg_video_stream_->channels()});
    return Status::OK();
  }
  Status Read(Tensor* value) { return ffmpeg_video_stream_->Read(value); }
  string DebugString() const override { return "FFmpegVideoReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  string filename_ TF_GUARDED_BY(mu_);
  int64 video_index_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
  std::unique_ptr<FFmpegVideoStream> ffmpeg_video_stream_ TF_GUARDED_BY(mu_);
  int64 frame_index_ TF_GUARDED_BY(mu_);
};

class FFmpegVideoReadableInitOp
    : public ResourceOpKernel<FFmpegVideoReadableResource> {
 public:
  explicit FFmpegVideoReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<FFmpegVideoReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<FFmpegVideoReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));

    OP_REQUIRES_OK(context, resource_->Init(input_tensor->scalar<tstring>()(),
                                            index_tensor->scalar<int64>()()));
  }
  Status CreateResource(FFmpegVideoReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new FFmpegVideoReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class FFmpegVideoReadableNextOp : public OpKernel {
 public:
  explicit FFmpegVideoReadableNextOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    FFmpegVideoReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* reset_tensor;
    OP_REQUIRES_OK(context, context->input("reset", &reset_tensor));
    bool reset = reset_tensor->scalar<bool>()();
    if (reset) {
      OP_REQUIRES_OK(context, resource->Seek(0));
    }

    TensorShape value_shape;
    OP_REQUIRES_OK(context, resource->Peek(&value_shape));
    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, value_shape, &value_tensor));
    if (value_shape.dim_size(0) > 0) {
      OP_REQUIRES_OK(context, resource->Read(value_tensor));
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>FfmpegAudioReadableInit").Device(DEVICE_CPU),
                        FFmpegAudioReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>FfmpegAudioReadableNext").Device(DEVICE_CPU),
                        FFmpegAudioReadableNextOp);

REGISTER_KERNEL_BUILDER(Name("IO>FfmpegVideoReadableInit").Device(DEVICE_CPU),
                        FFmpegVideoReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>FfmpegVideoReadableNext").Device(DEVICE_CPU),
                        FFmpegVideoReadableNextOp);

class FFmpegDecodeVideoOp : public OpKernel {
 public:
  explicit FFmpegDecodeVideoOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));

    string input = input_tensor->scalar<tstring>()();
    SizedRandomAccessFile file(env_, "memory", input.data(), input.size());

    FFmpegInit();

    FFmpegVideoStream stream("memory", &file, input.size());
    OP_REQUIRES_OK(context, stream.OpenVideo(index_tensor->scalar<int64>()()));
    int64 frames = 0;
    OP_REQUIRES_OK(context, stream.PeekAll(&frames));

    Tensor* video_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0,
                       TensorShape({frames, stream.height(), stream.width(),
                                    stream.channels()}),
                       &video_tensor));

    OP_REQUIRES_OK(context, stream.Read(video_tensor));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>FfmpegDecodeVideo").Device(DEVICE_CPU),
                        FFmpegDecodeVideoOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
