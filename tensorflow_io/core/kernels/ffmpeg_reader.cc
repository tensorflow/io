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

#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/file_system.h"


extern "C" {

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libavutil/log.h"
#include "libswscale/swscale.h"
#include <dlfcn.h>

}

#include "kernels/ffmpeg_reader.h"

namespace tensorflow {
namespace data {

static mutex mu(LINKER_INITIALIZED);
static unsigned count(0);
void FFmpegReaderInit() {
  mutex_lock lock(mu);
  count++;
  if (count == 1) {
    // Set log level if needed
    static const struct { const char *name; int level; } log_levels[] = {
        { "quiet"  , AV_LOG_QUIET   },
        { "panic"  , AV_LOG_PANIC   },
        { "fatal"  , AV_LOG_FATAL   },
        { "error"  , AV_LOG_ERROR   },
        { "warning", AV_LOG_WARNING },
        { "info"   , AV_LOG_INFO    },
        { "verbose", AV_LOG_VERBOSE },
        { "debug"  , AV_LOG_DEBUG   },
        // { "trace"  , AV_LOG_TRACE   },
    };
    const char* log_level_name = getenv("FFMPEG_LOG_LEVEL");
    if (log_level_name != nullptr) {
      string log_level = log_level_name;
      for (size_t i = 0; i < sizeof(log_levels)/sizeof(log_levels[0]); i++) {
        if (log_level == log_levels[i].name) {
          LOG(INFO) << "FFmpeg log level: " << log_level;
          av_log_set_level(log_levels[i].level);
          break;
        }
      }
    }

    // Register all formats and codecs
    av_register_all();
  }
}

static int io_read_packet(void *opaque, uint8_t *buf, int buf_size) {
  FFmpegReader *r = (FFmpegReader *)opaque;
  StringPiece result;
  Status status = r->stream_->Read(r->offset_, buf_size, &result, (char *)buf);
  if (!(status.ok() || errors::IsOutOfRange(status))) {
    return -1;
  }
  r->offset_ += result.size();
  return result.size();
}

static int64_t io_seek(void *opaque, int64_t offset, int whence) {
  FFmpegReader *r = (FFmpegReader *)opaque;
  uint64 file_size = 0;
  Status status = r->stream_->GetFileSize(&file_size);
  if (!status.ok()) {
    return -1;
  }
  switch (whence)
  {
  case SEEK_SET:
    if (offset > file_size) {
      return -1;
    }
    r->offset_ = offset;
    return r->offset_;
  case SEEK_CUR:
    if (r->offset_ + offset > file_size) {
      return -1;
    }
    r->offset_ += offset;
    return r->offset_;
  case SEEK_END:
    if (offset > file_size) {
      return -1;
    }
    r->offset_ = file_size - offset;
    return r->offset_;
  case AVSEEK_SIZE:
    return file_size;
  default:
    break;
  }
  return -1;
}


Status FFmpegReader::InitializeReader()
{
    // Allocate format
    if ((format_context_ = avformat_alloc_context()) == NULL) {
      return errors::InvalidArgument("could not allocate format context");
    }
    // Allocate context
    if ((io_context_ = avio_alloc_context(NULL, 0, 0, this, io_read_packet, NULL, io_seek)) == NULL) {
      return errors::InvalidArgument("could not allocate io context");
    }
    format_context_->pb = io_context_;
    // Open input file, and allocate format context
    if (avformat_open_input(&format_context_, filename_.c_str(), NULL, NULL) < 0) {
      return errors::InvalidArgument("could not open media file: ", filename_);
    }
    // Retrieve stream information
    if (avformat_find_stream_info(format_context_, NULL) < 0) {
      return errors::InvalidArgument("could not find stream information: ", filename_);
    }
    // Find media stream
    if ((stream_index_ = av_find_best_stream(format_context_, MediaType(), -1, -1, NULL, 0)) < 0) {
      return errors::InvalidArgument("could not find media stream: ", filename_);
    }

    AVStream *media_stream = format_context_->streams[stream_index_];
#if LIBAVCODEC_VERSION_MAJOR > 56
    // Find decoder for the stream
    AVCodec *codec = avcodec_find_decoder(media_stream->codecpar->codec_id);
    if (!codec) {
      return errors::Internal("could not find media codec: ", media_stream->codecpar->codec_id);
    }
    // Allocate a codec context for the decoder
    codec_context_ = avcodec_alloc_context3(codec);
    if (!codec_context_) {
      return errors::Internal("could not allocate codec context");
    }
    // Copy codec parameters from input stream to output codec context
    if (avcodec_parameters_to_context(codec_context_, media_stream->codecpar) < 0) {
      return errors::Internal("could not copy codec parameters from input stream to output codec context");
    }
#else
    codec_context_ = media_stream->codec;
    // Find decoder for the stream
    AVCodec *codec = avcodec_find_decoder(codec_context_->codec_id);
    if (!codec) {
      return errors::Internal("could not find media codec: ", codec_context_->codec_id);
    }
#endif
    // Initialize the decoders
    // TODO (yongtang): avcodec_open2 is not thread-safe
    AVDictionary *opts = NULL;
    if (avcodec_open2(codec_context_, codec, &opts) < 0) {
      return errors::Internal("could not open codec");
    }

    // Allocate frame
#if LIBAVCODEC_VERSION_MAJOR > 54
    frame_ = av_frame_alloc();
#else
    frame_ = avcodec_alloc_frame();
#endif
    if (!frame_) {
      return errors::Internal("could not allocate frame");
    }

    // Initialize packet
    av_init_packet(&packet_);
    packet_.data = NULL;
    packet_.size = 0;

    return Status::OK();
}

bool FFmpegReader::ReadAhead(bool first)
{
    while (packet_more_ || frame_more_) {
      while (packet_more_) {
        packet_more_ = false;
	if (packet_.stream_index == stream_index_) {
          int got_frame = 0;
          int decoded = DecodeFrame(&got_frame);
          if (!frame_more_ && got_frame) {
            // This is the cached packet.
            ProcessFrame();
            packet_more_ = true;
            return true;
          }
          if (decoded >= 0 && got_frame) {
            ProcessFrame();
	    if (packet_.data) {
	      packet_.data += decoded;
	      packet_.size -= decoded;
              packet_more_ = (packet_.size > 0);
	    }
            return true;
          }
	}
      }
      if (frame_more_) {
        // If this is not the first time, unref the packet
#if LIBAVCODEC_VERSION_MAJOR > 54
        av_packet_unref(&packet_);
#else
	// NOTE: libav 9.20 does not need unref or free here.
	// av_packet_unref(&packet_);
#endif
	frame_more_ = (av_read_frame(format_context_, &packet_) == 0); 
	if (!frame_more_) {
          // Flush out the cached packet
	  packet_more_ = true;
          packet_.data = NULL;
          packet_.size = 0;
	} else {
	  // More packet to process
          packet_more_ = true;
	}
      }
    }
    return false;
}

FFmpegReader::~FFmpegReader() {
#if LIBAVCODEC_VERSION_MAJOR > 54
    av_frame_free(&frame_);
#else
    avcodec_free_frame(&frame_);
#endif
#if LIBAVCODEC_VERSION_MAJOR > 56
    avcodec_free_context(&codec_context_);
#endif
    avformat_close_input(&format_context_);
    av_free(format_context_);
    if (io_context_ != NULL) {
      av_free(io_context_);
    }
}

}  // namespace data
}  // namespace tensorflow
