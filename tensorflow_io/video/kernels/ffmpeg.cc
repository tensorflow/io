/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/file_system.h"


extern "C" {

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
#include <dlfcn.h>

}

#include "kernels/video_reader.h"

namespace tensorflow {
namespace data {
namespace video {

static int io_read_packet(void *opaque, uint8_t *buf, int buf_size) {
  VideoReader *r = (VideoReader *)opaque;
  StringPiece result;
  Status status = r->stream_->Read(r->offset_, buf_size, &result, (char *)buf);
  if (!(status.ok() || errors::IsOutOfRange(status))) {
    return -1;
  }
  r->offset_ += result.size();
  return result.size();
}

static int64_t io_seek(void *opaque, int64_t offset, int whence) {
  VideoReader *r = (VideoReader *)opaque;
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

Status VideoReader::ReadHeader()
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
      return errors::InvalidArgument("could not open video file: ", filename_);
    }
    // Retrieve stream information
    if (avformat_find_stream_info(format_context_, NULL) < 0) {
      return errors::InvalidArgument("could not find stream information: ", filename_);
    }
    // Find video stream
    if ((stream_index_ = av_find_best_stream(format_context_, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0)) < 0) {
      return errors::InvalidArgument("could not find video stream: ", filename_);
    }

    AVStream *video_stream = format_context_->streams[stream_index_];
#if LIBAVCODEC_VERSION_MAJOR > 56
    // Find decoder for the stream
    AVCodec *codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
    if (!codec) {
      return errors::Internal("could not find video codec: ", video_stream->codecpar->codec_id);
    }
    // Allocate a codec context for the decoder
    codec_context_ = avcodec_alloc_context3(codec);
    if (!codec_context_) {
      return errors::Internal("could not allocate codec context");
    }
    // Copy codec parameters from input stream to output codec context
    if (avcodec_parameters_to_context(codec_context_, video_stream->codecpar) < 0) {
      return errors::Internal("could not copy codec parameters from input stream to output codec context");
    }
#else
    codec_context_ = video_stream->codec;
    // Find decoder for the stream
    AVCodec *codec = avcodec_find_decoder(codec_context_->codec_id);
    if (!codec) {
      return errors::Internal("could not find video codec: ", codec_context_->codec_id);
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

    // create scaling context
#if LIBSWSCALE_VERSION_MAJOR > 2
    sws_context_ = sws_getContext(codec_context_->width, codec_context_->height, codec_context_->pix_fmt, codec_context_->width, codec_context_->height, AV_PIX_FMT_RGB24, 0, NULL, NULL, NULL);
#else
    sws_context_ = sws_getContext(codec_context_->width, codec_context_->height, codec_context_->pix_fmt, codec_context_->width, codec_context_->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
#endif
    if (!sws_context_) {
      return errors::Internal("could not allocate sws context");
    }
#if LIBAVCODEC_VERSION_MAJOR > 54
    frame_rgb_ = av_frame_alloc();
#else
    frame_rgb_ = avcodec_alloc_frame();
#endif
    if (!frame_rgb_) {
      return errors::Internal("could not allocate rgb frame");
    }
    // Determine required buffer size and allocate buffer
#if LIBAVCODEC_VERSION_MAJOR > 54
    num_bytes_ = av_image_get_buffer_size(AV_PIX_FMT_RGB24, codec_context_->width, codec_context_->height, 1);
#else
    num_bytes_ = avpicture_get_size(AV_PIX_FMT_RGB24, codec_context_->width, codec_context_->height);
#endif
    buffer_rgb_ = (uint8_t *)av_malloc(num_bytes_ * sizeof(uint8_t));
    avpicture_fill((AVPicture *)frame_rgb_, buffer_rgb_, AV_PIX_FMT_RGB24, codec_context_->width, codec_context_->height);

    frame_more_ = true;
    packet_more_ = false;
    buffer_more_ = ReadAhead(true);

    return Status::OK();
  }

bool VideoReader::ReadAhead(bool first)
{
    while (packet_more_ || frame_more_) {
      while (packet_more_) {
        packet_more_ = false;
	if (packet_.stream_index == stream_index_) {
          int got_frame = 0;
          int decoded = avcodec_decode_video2(codec_context_, frame_, &got_frame, &packet_);
          if (!frame_more_ && got_frame) {
            // This is the cached packet.
            sws_scale(sws_context_, frame_->data, frame_->linesize, 0, codec_context_->height, frame_rgb_->data, frame_rgb_->linesize);
            packet_more_ = true;
            return true;
          }
          if (decoded >= 0 && got_frame) {
	    sws_scale(sws_context_, frame_->data, frame_->linesize, 0, codec_context_->height, frame_rgb_->data, frame_rgb_->linesize);
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

Status VideoReader::ReadFrame(int *num_bytes, uint8_t**value, int *height, int *width)
{
    *height = codec_context_->height;
    *width = codec_context_->width;
    *num_bytes = num_bytes_;
    if (buffer_more_) {
      *value = buffer_rgb_;
      buffer_more_ = ReadAhead(true);
      return Status::OK();
    }
    return errors::OutOfRange("EOF");
}

VideoReader::~VideoReader() {
    av_free(buffer_rgb_);
#if LIBAVCODEC_VERSION_MAJOR > 54
    av_frame_free(&frame_rgb_);
#else
    avcodec_free_frame(&frame_rgb_);
#endif
    sws_freeContext(sws_context_);
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

}  // namespace
}  // namespace data
}  // namespace tensorflow
