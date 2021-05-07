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

mutex mu(LINKER_INITIALIZED);

}  // namespace

namespace data {

void FFmpegInit();

class FFmpegReadStream {
 public:
  FFmpegReadStream(const string& filename, SizedRandomAccessFile* file,
                   uint64 file_size)
      : filename_(filename),
        file_(file),
        file_size_(file_size),
        offset_(0),
        format_context_(nullptr,
                        [](AVFormatContext* p) {
                          if (p != nullptr) {
                            avformat_close_input(&p);
                            avformat_free_context(p);
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
        stream_index_(-1) {
  }
  virtual ~FFmpegReadStream() {}

  int64 Streams() { return format_context_.get()->nb_streams; }
  int64 StreamType(int64 stream_index) {
#if LIBAVCODEC_VERSION_MAJOR > 56
    int media_type =
        format_context_->streams[stream_index]->codecpar->codec_type;
#else
    int media_type = format_context_->streams[stream_index]->codec->codec_type;
#endif
    return media_type;
  }
  virtual Status Open(int64 stream_index) {
    offset_ = 0;
    AVFormatContext* format_context;
    if ((format_context = avformat_alloc_context()) != NULL) {
      AVIOContext* io_context;
      if ((io_context = avio_alloc_context(NULL, 0, 0, this,
                                           FFmpegReadStream::ReadPacket, NULL,
                                           FFmpegReadStream::Seek)) != NULL) {
        format_context->pb = io_context;
        if (avformat_open_input(&format_context, filename_.c_str(), NULL,
                                NULL) >= 0) {
          if (avformat_find_stream_info(format_context, NULL) >= 0) {
            // No plan to read any other stream frame
            for (int64 i = 0; i < format_context->nb_streams; i++) {
              if (stream_index != i) {
                format_context->streams[i]->discard = AVDISCARD_ALL;
              }
            }
            stream_index_ = stream_index;
            io_context_.reset(io_context);
            format_context_.reset(format_context);
            return Status::OK();
          }
          avformat_close_input(&format_context);
        }
        av_free(io_context->buffer);
#if LIBAVCODEC_VERSION_MAJOR > 56
        avio_context_free(&io_context);
#else
        av_free(io_context);
#endif
      }
      avformat_free_context(format_context);
    }
    return errors::InvalidArgument("unable to open file: ", filename_);
  }

  static int ReadPacket(void* opaque, uint8_t* buf, int buf_size) {
    FFmpegReadStream* r = (FFmpegReadStream*)opaque;
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
    FFmpegReadStream* r = (FFmpegReadStream*)opaque;
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
};

class FFmpegReadStreamMeta : public FFmpegReadStream {
 public:
  FFmpegReadStreamMeta(const string& filename, SizedRandomAccessFile* file,
                       uint64 file_size, int64 media_type)
      : FFmpegReadStream(filename, file, file_size),
        media_type_(media_type),
        record_index_(0),
        nb_frames_(-1),
        dtype_(DT_INVALID),
        packet_scope_(nullptr,
                      [](AVPacket* p) {
                        if (p != nullptr) {
                          av_packet_unref(p);
                        }
                      }),
        codec_context_scope_(nullptr,
                             [](AVCodecContext* p) {
                               if (p != nullptr) {
                                 avcodec_free_context(&p);
                               }
                             }),
        initialized_(false) {}

  virtual ~FFmpegReadStreamMeta() {}

  virtual Status Open(int64 stream_index) override {
    record_index_ = 0;
    initialized_ = false;
    TF_RETURN_IF_ERROR(FFmpegReadStream::Open(stream_index));
    if (StreamType(stream_index) != media_type_) {
      return errors::Internal("type mismatch: ", StreamType(stream_index),
                              " vs. ", media_type_);
    }
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
      AVDictionary* opts = NULL;
      mutex_lock lock(mu);
      if (avcodec_open2(codec_context_, codec, &opts) < 0) {
        return errors::Internal("could not open codec");
      }
    }

    nb_frames_ = format_context_->streams[stream_index]->nb_frames;

    return Status::OK();
  }
  Status InitializeDecoder() {
    // Initialize the decoders

    samples_index_ = 0;
    // Read first frame if possible
    av_init_packet(&packet_);
    packet_.data = NULL;
    packet_.size = 0;
    // TODO: reference after first?
    packet_scope_.reset(&packet_);

    return Status::OK();
  }
  int64 Type() { return media_type_; }
  int64 RecordIndex() { return record_index_; }
  int64 Frames() { return nb_frames_; }
  string Codec() { return codec_; }
  PartialTensorShape Shape() { return shape_; }
  DataType DType() { return dtype_; }
  Status DecodePacket() {
    if (packet_scope_.get() == nullptr) {
      return errors::OutOfRange("EOF reached");
    }
    int ret;
    av_init_packet(&packet_);
    do {
      ret = av_read_frame(format_context_.get(), &packet_);
      if (ret < 0) {
        break;
      }
      if (packet_.stream_index != stream_index_) {
        av_packet_unref(&packet_);
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
  virtual Status DecodeFrame(int* got_frame) = 0;
  virtual Status ReadDecoded(int64 record_to_read, int64* record_read,
                             Tensor* value) = 0;
  virtual Status Read(int64 record_to_read, int64* record_read, Tensor* value) {
    if (!initialized_) {
      TF_RETURN_IF_ERROR(InitializeDecoder());
      TF_RETURN_IF_ERROR(DecodePacket());
      initialized_ = true;
    }
    *record_read = 0;
    Status status;
    do {
      TF_RETURN_IF_ERROR(ReadDecoded(record_to_read, record_read, value));
      if ((*record_read) >= record_to_read) {
        record_index_ += (*record_read);
        return Status::OK();
      }
      status = DecodePacket();
    } while (status.ok());
    TF_RETURN_IF_ERROR(ReadDecoded(record_to_read, record_read, value));
    record_index_ += (*record_read);
    return Status::OK();
  }

 protected:
  int64 media_type_;
  int64 record_index_;
  int64 nb_frames_;
  PartialTensorShape shape_;
  DataType dtype_;
  string codec_;
  AVPacket packet_;
  std::unique_ptr<AVPacket, void (*)(AVPacket*)> packet_scope_;
  AVCodecContext* codec_context_;
  std::unique_ptr<AVCodecContext, void (*)(AVCodecContext*)>
      codec_context_scope_;
  std::deque<std::unique_ptr<AVFrame, void (*)(AVFrame*)>> frames_;
  bool initialized_ = false;
  int got_frame_;
  int64 samples_index_;
};

class FFmpegVideoReadStreamMeta : public FFmpegReadStreamMeta {
 public:
  FFmpegVideoReadStreamMeta(const string& filename, SizedRandomAccessFile* file,
                            uint64 file_size)
      : FFmpegReadStreamMeta(filename, file, file_size, AVMEDIA_TYPE_VIDEO),
        height_(-1),
        width_(-1),
        num_bytes_(-1),
        sws_context_(nullptr, [](SwsContext* p) {
          if (p != nullptr) {
            sws_freeContext(p);
          }
        }) {}
  virtual ~FFmpegVideoReadStreamMeta() {}
  virtual Status Open(int64 stream_index) override {
    TF_RETURN_IF_ERROR(FFmpegReadStreamMeta::Open(stream_index));

    height_ = codec_context_->height;
    width_ = codec_context_->width;
    num_bytes_ = av_image_get_buffer_size(
        AV_PIX_FMT_RGB24, codec_context_->width, codec_context_->height, 1);

    SwsContext* sws_context = sws_getContext(
        codec_context_->width, codec_context_->height, codec_context_->pix_fmt,
        codec_context_->width, codec_context_->height, AV_PIX_FMT_RGB24, 0,
        NULL, NULL, NULL);
    if (!sws_context) {
      return errors::Internal("could not allocate sws context");
    }
    sws_context_.reset(sws_context);

    shape_ = PartialTensorShape({-1, height_, width_, 3});
    dtype_ = DT_UINT8;
    return Status::OK();
  }
  Status ReadDecoded(int64 record_to_read, int64* record_read,
                     Tensor* value) override {
    while ((*record_read) < record_to_read) {
      if (frames_.empty()) {
        return Status::OK();
      }
      int64 offset = (*record_read) * height_ * width_ * 3;
      memcpy(reinterpret_cast<char*>(&value->flat<uint8>().data()[offset]),
             reinterpret_cast<char*>(frames_buffer_.front().get()),
             num_bytes_ * sizeof(uint8_t));
      frames_.pop_front();
      frames_buffer_.pop_front();
      (*record_read)++;
    }
    return Status::OK();
  }
  Status DecodeFrame(int* got_frame) override {
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
      std::unique_ptr<AVFrame, void (*)(AVFrame*)> frame_rgb(
          av_frame_alloc(), [](AVFrame* p) {
            if (p != nullptr) {
              av_frame_free(&p);
            }
          });
      std::unique_ptr<uint8_t, void (*)(uint8_t*)> buffer_rgb(
          (uint8_t*)av_malloc(num_bytes_ * sizeof(uint8_t)), [](uint8_t* p) {
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
  virtual Status Peek(int64* record_to_read) {
    if (!initialized_) {
      TF_RETURN_IF_ERROR(InitializeDecoder());
      TF_RETURN_IF_ERROR(DecodePacket());
      initialized_ = true;
    }
    Status status;
    do {
      status = DecodePacket();
    } while (status.ok());
    *record_to_read = frames_.size();
    return Status::OK();
  }

  int64 Height() { return height_; }
  int64 Width() { return width_; }

 private:
  int64 height_;
  int64 width_;
  int64 num_bytes_;
  std::deque<std::unique_ptr<uint8_t, void (*)(uint8_t*)>> frames_buffer_;
  std::unique_ptr<SwsContext, void (*)(SwsContext*)> sws_context_;
};

class FFmpegAudioReadStreamMeta : public FFmpegReadStreamMeta {
 public:
  FFmpegAudioReadStreamMeta(const string& filename, SizedRandomAccessFile* file,
                            uint64 file_size)
      : FFmpegReadStreamMeta(filename, file, file_size, AVMEDIA_TYPE_AUDIO),
        channels_(-1),
        rate_(-1) {}
  virtual ~FFmpegAudioReadStreamMeta() {}

  virtual Status Open(int64 stream_index) override {
    TF_RETURN_IF_ERROR(FFmpegReadStreamMeta::Open(stream_index));
#if LIBAVCODEC_VERSION_MAJOR > 56
    int format = format_context_->streams[stream_index]->codecpar->format;
    channels_ = format_context_->streams[stream_index]->codecpar->channels;
    rate_ = format_context_->streams[stream_index]->codecpar->sample_rate;
#else
    int format = format_context_->streams[stream_index]->codec->sample_fmt;
    channels_ = format_context_->streams[stream_index]->codec->channels;
    rate_ = format_context_->streams[stream_index]->codec->sample_rate;
#endif
    shape_ = PartialTensorShape({-1, channels_});
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

      case AV_SAMPLE_FMT_U8P:  ///< unsigned 8 bits, planar
        dtype_ = DT_UINT8;
        break;
      case AV_SAMPLE_FMT_S16P:  ///< signed 16 bits, planar
        dtype_ = DT_INT16;
        break;
      case AV_SAMPLE_FMT_S32P:  ///< signed 32 bits, planar
        dtype_ = DT_INT32;
        break;
      case AV_SAMPLE_FMT_FLTP:  ///< float, planar
        dtype_ = DT_FLOAT;
        break;
      case AV_SAMPLE_FMT_DBLP:  ///< double, planar
        dtype_ = DT_DOUBLE;
        break;
      // case AV_SAMPLE_FMT_S64:         ///< signed 64 bits
      // case AV_SAMPLE_FMT_S64P:        ///< signed 64 bits, planar
      default:
        return errors::InvalidArgument("invalid audio (", stream_index,
                                       ") format: ", format);
    }

    return Status::OK();
  }
  Status ReadDecodedRecord(int64 record_to_read, int64* record_read,
                           Tensor* value) {
    int64 datasize = av_get_bytes_per_sample(codec_context_->sample_fmt);
    if (datasize != DataTypeSize(dtype_)) {
      return errors::InvalidArgument("failed to calculate data size");
    }
    char* base;
    switch (dtype_) {
      case DT_INT16:
        base = ((char*)(value->flat<int16>().data()));
        break;
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       DataTypeString(dtype_));
    }
    while (samples_index_ < frames_.front()->nb_samples) {
      for (int64 channel = 0; channel < codec_context_->channels; channel++) {
        char* data =
            base +
            datasize * ((*record_read) * codec_context_->channels + channel);
        char* copy = ((char*)(frames_.front()->data[channel])) +
                     datasize * samples_index_;
        memcpy(data, copy, datasize);
      }
      (*record_read)++;
      samples_index_++;
      if ((*record_read) >= record_to_read) {
        return Status::OK();
      }
    }
    return Status::OK();
  }
  Status ReadDecoded(int64 record_to_read, int64* record_read,
                     Tensor* value) override {
    while ((*record_read) < record_to_read) {
      if (frames_.empty()) {
        return Status::OK();
      }
      if (samples_index_ < frames_.front()->nb_samples) {
        TF_RETURN_IF_ERROR(
            ReadDecodedRecord(record_to_read, record_read, value));
      }
      if (!frames_.empty() && samples_index_ >= frames_.front()->nb_samples) {
        frames_.pop_front();
        samples_index_ = 0;
      }
    }
    return Status::OK();
  }
  Status DecodeFrame(int* got_frame) override {
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
  int64 Channels() { return channels_; }
  int64 Rate() { return rate_; }

 private:
  int64 channels_;
  int64 rate_;
};

class FFmpegSubtitleReadStreamMeta : public FFmpegReadStreamMeta {
 public:
  FFmpegSubtitleReadStreamMeta(const string& filename,
                               SizedRandomAccessFile* file, uint64 file_size)
      : FFmpegReadStreamMeta(filename, file, file_size, AVMEDIA_TYPE_SUBTITLE) {
  }
  virtual ~FFmpegSubtitleReadStreamMeta() {}
  virtual Status Open(int64 stream_index) override {
    TF_RETURN_IF_ERROR(FFmpegReadStreamMeta::Open(stream_index));
    shape_ = PartialTensorShape({-1});
    dtype_ = DT_STRING;
    return Status::OK();
  }

 private:
  Status ReadDecoded(int64 record_to_read, int64* record_read,
                     Tensor* value) override {
    while ((*record_read) < record_to_read) {
      if (subtitles_.empty()) {
        return Status::OK();
      }
      value->flat<tstring>()((*record_read)) = subtitles_.front();
      subtitles_.pop_front();
      (*record_read)++;
    }
    return Status::OK();
  }
  Status DecodeFrame(int* got_frame) override {
    AVSubtitle subtitle;
    int decoded = avcodec_decode_subtitle2(codec_context_, &subtitle, got_frame,
                                           &packet_);
    if (decoded < 0) {
      return errors::InvalidArgument("error decoding subtitle frame (", decoded,
                                     ")");
    }
    decoded = FFMIN(decoded, packet_.size);
    packet_.data += decoded;
    packet_.size -= decoded;
    if (*got_frame) {
      // We expect one rect
      if (subtitle.num_rects != 1) {
        return errors::InvalidArgument(
            "number of rects has to be 1, received: ", subtitle.num_rects);
      }
      switch (subtitle.rects[0]->type) {
        case SUBTITLE_ASS:
          if (!strncmp(subtitle.rects[0]->ass, "Dialogue: ", 10)) {
            string buffer = string(subtitle.rects[0]->ass);
            // find after 9th ","
            size_t position = 0;
            for (int64 i = 0; i < 9; i++) {
              position = buffer.find(",", position);
              if (position == string::npos) {
                return errors::InvalidArgument("invalid libass format: ",
                                               buffer);
              }
              position++;
            }
            subtitles_.push_back(buffer.substr(position));
          } else {
            subtitles_.push_back(string(subtitle.rects[0]->ass));
          }
          break;
        case SUBTITLE_TEXT:
          subtitles_.push_back(string(subtitle.rects[0]->text));
          break;
        default:
          return errors::InvalidArgument("unsupported subtitle type: ",
                                         subtitle.rects[0]->type);
      }
    }
    return Status::OK();
  }
  std::deque<string> subtitles_;
};

class FFmpegReadable : public IOReadableInterface {
 public:
  FFmpegReadable(Env* env) : env_(env) {}

  virtual ~FFmpegReadable() {}
  Status Init(const std::vector<string>& input,
              const std::vector<string>& metadata, const void* memory_data,
              const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 filename is not supported");
    }
    const string& filename = input[0];
    file_.reset(
        new SizedRandomAccessFile(env_, filename, memory_data, memory_size));
    TF_RETURN_IF_ERROR(env_->GetFileSize(filename, &file_size_));
    ffmpeg_file_.reset(new FFmpegReadStream(filename, file_.get(), file_size_));

    FFmpegInit();
    TF_RETURN_IF_ERROR(ffmpeg_file_->Open(-1));

    int64 audio_index = 0, video_index = 0, subtitle_index = 0;
    for (int64 i = 0; i < ffmpeg_file_->Streams(); i++) {
      switch (ffmpeg_file_->StreamType(i)) {
        case AVMEDIA_TYPE_VIDEO:
          columns_meta_.push_back(std::unique_ptr<FFmpegReadStreamMeta>(
              new FFmpegVideoReadStreamMeta(filename, file_.get(),
                                            file_size_)));
          TF_RETURN_IF_ERROR(columns_meta_[i]->Open(i));
          shapes_.push_back(columns_meta_[i]->Shape());
          dtypes_.push_back(columns_meta_[i]->DType());
          columns_.push_back(absl::StrCat("v:", video_index));
          columns_index_[columns_.back()] = i;
          video_index++;
          break;
        case AVMEDIA_TYPE_AUDIO:
          columns_meta_.push_back(std::unique_ptr<FFmpegReadStreamMeta>(
              new FFmpegAudioReadStreamMeta(filename, file_.get(),
                                            file_size_)));
          TF_RETURN_IF_ERROR(columns_meta_[i]->Open(i));
          shapes_.push_back(columns_meta_[i]->Shape());
          dtypes_.push_back(columns_meta_[i]->DType());
          columns_.push_back(absl::StrCat("a:", audio_index));
          columns_index_[columns_.back()] = i;
          audio_index++;
          break;
        case AVMEDIA_TYPE_SUBTITLE:
          columns_meta_.push_back(std::unique_ptr<FFmpegReadStreamMeta>(
              new FFmpegSubtitleReadStreamMeta(filename, file_.get(),
                                               file_size_)));
          TF_RETURN_IF_ERROR(columns_meta_[i]->Open(i));
          shapes_.push_back(columns_meta_[i]->Shape());
          dtypes_.push_back(columns_meta_[i]->DType());
          columns_.push_back(absl::StrCat("s:", subtitle_index));
          columns_index_[columns_.back()] = i;
          subtitle_index++;
          break;
        default:
          return errors::InvalidArgument(
              "invalid steam (", i, ") type: ", ffmpeg_file_->StreamType(i));
      }
    }
    return Status::OK();
  }
  Status Components(std::vector<string>* components) override {
    components->clear();
    for (size_t i = 0; i < columns_.size(); i++) {
      components->push_back(columns_[i]);
    }
    return Status::OK();
  }
  Status Spec(const string& component, PartialTensorShape* shape,
              DataType* dtype, bool label) override {
    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];
    *shape = shapes_[column_index];
    *dtype = dtypes_[column_index];
    return Status::OK();
  }

  Status Extra(const string& component, std::vector<Tensor>* extra) override {
    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];
    // Expose a sample `rate`
    FFmpegAudioReadStreamMeta* meta = dynamic_cast<FFmpegAudioReadStreamMeta*>(
        columns_meta_[column_index].get());
    Tensor rate(DT_INT64, TensorShape({}));
    rate.scalar<int64>()() = (meta != nullptr) ? meta->Rate() : 0;
    extra->push_back(rate);
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop, const string& component,
              int64* record_read, Tensor* value, Tensor* label) override {
    *record_read = 0;
    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];
    if (start != columns_meta_[column_index]->RecordIndex()) {
      // If we reach to the end, then just 0
      if (start > columns_meta_[column_index]->RecordIndex()) {
        return Status::OK();
      }
      if (start != 0) {
        return errors::InvalidArgument(
            "ffmepg dataset could not seek to a random location");
      }
      // Else we will recreate the meta.
      TF_RETURN_IF_ERROR(columns_meta_[column_index]->Open(column_index));
    }
    return columns_meta_[column_index]->Read(stop - start, record_read, value);
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("FFmpegReadable");
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
  std::unique_ptr<FFmpegReadStream> ffmpeg_file_ TF_GUARDED_BY(mu_);

  std::vector<DataType> dtypes_;
  std::vector<PartialTensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
  std::vector<std::unique_ptr<FFmpegReadStreamMeta>> columns_meta_;
};

REGISTER_KERNEL_BUILDER(Name("IO>FfmpegReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<FFmpegReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>FfmpegReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<FFmpegReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>FfmpegReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<FFmpegReadable>);

}  // namespace data
}  // namespace tensorflow
