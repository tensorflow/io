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

    for (int64 i = 0; i < format_context_->nb_streams; i++)
    {
#if LIBAVCODEC_VERSION_MAJOR > 56
      int media_type = format_context_->streams[i]->codecpar->codec_type;
#else
      int media_type = format_context_->streams[i]->codec->codec_type;
#endif
      switch (media_type)
      {
      case AVMEDIA_TYPE_VIDEO: {
        // n * h * w * 3 (uint8) / always RGB
#if LIBAVCODEC_VERSION_MAJOR > 56
        shapes_.push_back(PartialTensorShape({-1, format_context_->streams[i]->codecpar->height, format_context_->streams[i]->codecpar->width, 3}));
#else
        shapes_.push_back(PartialTensorShape({-1, format_context_->streams[i]->codec->height, format_context_->streams[i]->codec->width, 3}));
#endif
        dtypes_.push_back(DT_UINT8);
        string column = absl::StrCat("video:", i);
        columns_.push_back(column);
        columns_index_[column] = i;
        }
        break;
      case AVMEDIA_TYPE_AUDIO: {
        DataType dtype = DT_INVALID;
// TODO: version check is not right for audio
#if LIBAVCODEC_VERSION_MAJOR > 56
        int format = format_context_->streams[i]->codecpar->format;
#else
        int format = format_context_->streams[i]->codec->sample_fmt;
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
        shapes_.push_back(PartialTensorShape({-1, format_context_->streams[i]->codec->channels}));
        dtypes_.push_back(dtype);
        string column = absl::StrCat("audio:", i);
        columns_.push_back(column);
        columns_index_[column] = i;
        }
        break;
      case AVMEDIA_TYPE_SUBTITLE:
      default:
        return errors::InvalidArgument("invalid stream (", i, ") type: ", media_type);
      };
    }

    return Status::OK();
  }
  Status Components(Tensor* components) override {
    *components = Tensor(DT_STRING, TensorShape({static_cast<int64>(columns_.size())}));
    for (size_t i = 0; i < columns_.size(); i++) {
      components->flat<string>()(i) = columns_[i];
    }
    return Status::OK();
  }
  Status Spec(const Tensor& component, PartialTensorShape* shape, DataType* dtype, bool label) override {
    if (columns_index_.find(component.scalar<string>()()) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component.scalar<string>()(), " is invalid");
    }
    int64 column_index = columns_index_[component.scalar<string>()()];
    *shape = shapes_[column_index];
    *dtype = dtypes_[column_index];
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop, const Tensor& component, Tensor* value, Tensor* label) override {
    int64 column_index = component.scalar<int64>()();

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
};

REGISTER_KERNEL_BUILDER(Name("FfmpegIndexableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<FFmpegIndexable>);
REGISTER_KERNEL_BUILDER(Name("FfmpegIndexableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<FFmpegIndexable>);
REGISTER_KERNEL_BUILDER(Name("FfmpegIndexableRead").Device(DEVICE_CPU),
                        IOIndexableReadOp<FFmpegIndexable>);
}  // namespace data
}  // namespace tensorflow
