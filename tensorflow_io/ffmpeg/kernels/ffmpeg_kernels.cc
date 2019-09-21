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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/io_stream.h"
#include "kernels/video_ffmpeg_reader.h"

namespace tensorflow {
namespace data {

class FFmpegReadStreamFile : public SizedRandomAccessFile {
 public:
  FFmpegReadStreamFile(Env* env, const string& filename, const void* optional_memory_buff, const size_t optional_memory_size)
  : SizedRandomAccessFile(env, filename, optional_memory_buff, optional_memory_size)
  , offset_(0)
  , file_size_(-1)
 {
    uint64 file_size = 0;
    Status status = GetFileSize(&file_size);
    file_size_ = (status.ok()) ? file_size : -1;
  }
  ~FFmpegReadStreamFile() {
  }

 public:
  int64 offset_ = 0;
  int64 file_size_ = -1;
};

class FFmpegReadStreamMeta {
 public:
  FFmpegReadStreamMeta(int64 media_type)
  : media_type_(media_type) {}
  virtual ~FFmpegReadStreamMeta() {}

  int64 Type() { return media_type_; }
 private:
  int64 media_type_;

};
class FFmpegVideoStreamMeta : public FFmpegReadStreamMeta {
 public:
  FFmpegVideoStreamMeta()
  : FFmpegReadStreamMeta(AVMEDIA_TYPE_VIDEO) {}
  virtual ~FFmpegVideoStreamMeta() {}

 private:
};

class FFmpegAudioStreamMeta : public FFmpegReadStreamMeta {
 public:
  FFmpegAudioStreamMeta(int64 rate)
  : FFmpegReadStreamMeta(AVMEDIA_TYPE_AUDIO)
  , rate_(rate) {}
  virtual ~FFmpegAudioStreamMeta() {}

  int64 Rate() { return rate_; }
 private:
  int64 rate_;
};

class FFmpegSubtitleStreamMeta : public FFmpegReadStreamMeta {
 public:
  FFmpegSubtitleStreamMeta(const std::vector<string>& entries)
  : FFmpegReadStreamMeta(AVMEDIA_TYPE_SUBTITLE)
  , entries_(entries) {}
  virtual ~FFmpegSubtitleStreamMeta() {}

  static Status Read(AVFormatContext *format_context, AVStream *media_stream, std::vector<string>* entries) {
    entries->clear();

#if LIBAVCODEC_VERSION_MAJOR > 56
    // Find decoder for the stream
    AVCodec *codec = avcodec_find_decoder(media_stream->codecpar->codec_id);
    if (!codec) {
      return errors::Internal("could not find media codec: ", media_stream->codecpar->codec_id);
    }
    // Allocate a codec context for the decoder
    AVCodecContext *codec_context = avcodec_alloc_context3(codec);
    if (!codec_context) {
      return errors::Internal("could not allocate codec context");
    }

    std::unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> codec_context_scope(codec_context, [](AVCodecContext* p) { if (p != nullptr) { avcodec_free_context(&p); } });

    // Copy codec parameters from input stream to output codec context
    if (avcodec_parameters_to_context(codec_context, media_stream->codecpar) < 0) {
      return errors::Internal("could not copy codec parameters from input stream to output codec context");
    }
#else
    AVCodecContext *codec_context = media_stream->codec;

    // Find decoder for the stream
    AVCodec *codec = avcodec_find_decoder(codec_context->codec_id);
    if (!codec) {
      return errors::Internal("could not find media codec: ", codec_context->codec_id);
    }
#endif
    // Initialize the decoders
    // TODO (yongtang): avcodec_open2 is not thread-safe
    AVDictionary *opts = NULL;
    if (avcodec_open2(codec_context, codec, &opts) < 0) {
      return errors::Internal("could not open codec");
    }

    // Initialize packet
    AVPacket packet;
    av_init_packet(&packet);
    packet.data = NULL;
    packet.size = 0;

    if (av_read_frame(format_context, &packet) < 0) {
      // No frame to read, assume empty;
      return Status::OK();
    }

    std::unique_ptr<AVPacket, void(*)(AVPacket*)> packet_scope(&packet, [](AVPacket* p) { if (p != nullptr) { av_packet_unref(p); } });
    do {
      if(packet.stream_index != media_stream->index) {
        continue;
      }
      AVSubtitle subtitle;
      int got_subtitle = 0;
      int decoded = avcodec_decode_subtitle2(codec_context, &subtitle, &got_subtitle, &packet);
      if (decoded >= 0 && got_subtitle > 0) {
        if (subtitle.num_rects > 1) {
          return errors::InvalidArgument("subtitle with more than 1 rect not supported: ", subtitle.num_rects);
        }
        string line;
        int rect_index = 0;
        if (subtitle.rects[rect_index]->type == SUBTITLE_TEXT) {
          line = string(subtitle.rects[rect_index]->text);
        } else if (subtitle.rects[rect_index]->type == SUBTITLE_ASS) {
          line = string(subtitle.rects[rect_index]->ass);
        } else {
          return errors::InvalidArgument("unsupported subtitle type: ", subtitle.rects[rect_index]->type);
        }
        packet.data += decoded;
        packet.size -= decoded;
        if (packet.size > 0) {
          return errors::InvalidArgument("packet is supposed to only have one subtitle, but still have ", packet.size, " bytes");
        }
        entries->push_back(line);
      }
    } while (av_read_frame(format_context, &packet) >= 0);

    return Status::OK();
  }
  
 private:
  std::vector<string> entries_;
};

static int IOFFmpegReadPacket(void *opaque, uint8_t *buf, int buf_size) {
  FFmpegReadStreamFile *r = (FFmpegReadStreamFile *)opaque;
  StringPiece result;
  Status status = r->Read(r->offset_, buf_size, &result, (char *)buf);
  if (!(status.ok() || errors::IsOutOfRange(status))) {
    return -1;
  }
  r->offset_ += result.size();
  return result.size();
}

static int64_t IOFFmpegSeek(void *opaque, int64_t offset, int whence) {
  FFmpegReadStreamFile *r = (FFmpegReadStreamFile *)opaque;
  switch (whence)
  {
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

class FFmpegIndexable : public IOIndexableInterface {
 public:
  FFmpegIndexable(Env* env)
  : env_(env) {}

  ~FFmpegIndexable() {
    avformat_close_input(&format_context_);
    av_free(format_context_);
    if (io_context_ != NULL) {
      av_free(io_context_);
    }
  }
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 filename is not supported");
    }
    const string& filename = input[0];
    file_.reset(new FFmpegReadStreamFile(env_, filename, memory_data, memory_size));
    if (file_->file_size_ < 0) {
      return errors::InvalidArgument("unable to find out the file size:", filename);
    }
    file_size_ = file_->file_size_;

    FFmpegReaderInit();

    // Allocate format
    if ((format_context_ = avformat_alloc_context()) == NULL) {
      return errors::InvalidArgument("could not allocate format context");
    }
    // Allocate context
    if ((io_context_ = avio_alloc_context(NULL, 0, 0, file_.get(), IOFFmpegReadPacket, NULL, IOFFmpegSeek)) == NULL) {
      return errors::InvalidArgument("could not allocate io context");
    }
    format_context_->pb = io_context_;
    // Open input file, and allocate format context
    if (avformat_open_input(&format_context_, filename.c_str(), NULL, NULL) < 0) {
      return errors::InvalidArgument("could not open media file: ", filename);
    }
    // Retrieve stream information
    if (avformat_find_stream_info(format_context_, NULL) < 0) {
      return errors::InvalidArgument("could not find stream information: ", filename);
    }

    int64 audio_index = 0, video_index = 0, subtitle_index = 0;
    for (int64 i = 0; i < format_context_->nb_streams; i++)
    {
#if LIBAVCODEC_VERSION_MAJOR > 56
      int media_type = format_context_->streams[i]->codecpar->codec_type;
      int64 height = format_context_->streams[i]->codecpar->height;
      int64 width = format_context_->streams[i]->codecpar->width;
#else
      int media_type = format_context_->streams[i]->codec->codec_type;
      int64 height = format_context_->streams[i]->codec->height;
      int64 width = format_context_->streams[i]->codec->width;
#endif
      int64 frames = format_context_->streams[i]->nb_frames;
      if (frames == 0) {
        // Set frames to -1 to have a proper shape of None
        frames = -1;
      }
      switch (media_type)
      {
      case AVMEDIA_TYPE_VIDEO: {
        // n * h * w * 3 (uint8) / always RGB
        shapes_.push_back(PartialTensorShape({frames, height, width, 3}));
        dtypes_.push_back(DT_UINT8);

        string column = absl::StrCat("v:", video_index);
        video_index++;
        columns_.push_back(column);
        columns_index_[column] = i;
        columns_meta_.push_back(std::unique_ptr<FFmpegReadStreamMeta>(new FFmpegVideoStreamMeta()));
        }
        break;
      case AVMEDIA_TYPE_AUDIO: {
        DataType dtype = DT_INVALID;
#if LIBAVCODEC_VERSION_MAJOR > 56
        int format = format_context_->streams[i]->codecpar->format;
        int64 channels = format_context_->streams[i]->codecpar->channels;
        int64 rate = format_context_->streams[i]->codecpar->sample_rate;
#else
        int format = format_context_->streams[i]->codec->sample_fmt;
        int64 channels = format_context_->streams[i]->codec->channels;
        int64 rate = format_context_->streams[i]->codec->sample_rate;
#endif
        switch (format)
        {
        case AV_SAMPLE_FMT_U8:          ///< unsigned 8 bits
          dtype = DT_UINT8;
          break;
        case AV_SAMPLE_FMT_S16:         ///< signed 16 bits
          dtype = DT_INT16;
          break;
        case AV_SAMPLE_FMT_S32:         ///< signed 32 bits
          dtype = DT_INT32;
          break;
        case AV_SAMPLE_FMT_FLT:         ///< float
          dtype = DT_FLOAT;
          break;
        case AV_SAMPLE_FMT_DBL:         ///< double
          dtype = DT_DOUBLE;
          break;

        case AV_SAMPLE_FMT_U8P:         ///< unsigned 8 bits, planar
          dtype = DT_UINT8;
          break;
        case AV_SAMPLE_FMT_S16P:        ///< signed 16 bits, planar
          dtype = DT_INT16;
          break;
        case AV_SAMPLE_FMT_S32P:        ///< signed 32 bits, planar
          dtype = DT_INT32;
          break;
        case AV_SAMPLE_FMT_FLTP:        ///< float, planar
          dtype = DT_FLOAT;
          break;
        case AV_SAMPLE_FMT_DBLP:        ///< double, planar
          dtype = DT_DOUBLE;
          break;
        // case AV_SAMPLE_FMT_S64:         ///< signed 64 bits
        // case AV_SAMPLE_FMT_S64P:        ///< signed 64 bits, planar
        default:
          return errors::InvalidArgument("invalid audio (", i, ") format: ", format);
        }
        // samples * channels
        shapes_.push_back(PartialTensorShape({frames, channels}));
        dtypes_.push_back(dtype);
        string column = absl::StrCat("a:", audio_index);
        audio_index++;
        columns_.push_back(column);
        columns_index_[column] = i;
        columns_meta_.push_back(std::unique_ptr<FFmpegReadStreamMeta>(new FFmpegAudioStreamMeta(rate)));
        }
        break;
      case AVMEDIA_TYPE_SUBTITLE: {
        // Subtitle always will not show frames.
        // Since it is expected to be small, let's just extract inline
        std::vector<string> entries;
        TF_RETURN_IF_ERROR(FFmpegSubtitleStreamMeta::Read(format_context_, format_context_->streams[i], &entries));
        if (entries.size() >= 0) {
          frames = static_cast<int64>(entries.size());
        }
        shapes_.push_back(PartialTensorShape({frames}));
        DataType dtype = DT_STRING;
        dtypes_.push_back(dtype);
        string column = absl::StrCat("s:", subtitle_index);
        subtitle_index++;
        columns_.push_back(column);
        columns_index_[column] = i;
        columns_meta_.push_back(std::unique_ptr<FFmpegReadStreamMeta>(new FFmpegSubtitleStreamMeta(entries)));
        }
        break;
      default:
        return errors::InvalidArgument("invalid stream (", i, ") type: ", media_type);
      };
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
  Status Spec(const string& component, PartialTensorShape* shape, DataType* dtype, bool label) override {
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
    Tensor rate(DT_INT32, TensorShape({}));
    FFmpegAudioStreamMeta *meta = dynamic_cast<FFmpegAudioStreamMeta *>(columns_meta_[column_index].get());
    rate.scalar<int32>()() = (meta != nullptr) ? meta->Rate() : 0;
    extra->push_back(rate);
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop, const string& component, Tensor* value, Tensor* label) override {
    return errors::Unimplemented("Read is currently not supported");
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("FFmpegIndexable");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<FFmpegReadStreamFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);

  AVFormatContext *format_context_ GUARDED_BY(mu_) = 0;
  AVIOContext *io_context_ GUARDED_BY(mu_) = NULL;

  std::vector<DataType> dtypes_;
  std::vector<PartialTensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
  std::vector<std::unique_ptr<FFmpegReadStreamMeta>> columns_meta_;
};

REGISTER_KERNEL_BUILDER(Name("FfmpegIndexableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<FFmpegIndexable>);
REGISTER_KERNEL_BUILDER(Name("FfmpegIndexableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<FFmpegIndexable>);
REGISTER_KERNEL_BUILDER(Name("FfmpegIndexableRead").Device(DEVICE_CPU),
                        IOIndexableReadOp<FFmpegIndexable>);
}  // namespace data
}  // namespace tensorflow
