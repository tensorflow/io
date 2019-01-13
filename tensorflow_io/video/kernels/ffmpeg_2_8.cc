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

Status VideoReader::ReadHeader()
{
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
    codec_context_ = video_stream->codec;
    // Find decoder for the stream
    AVCodec *codec = avcodec_find_decoder(codec_context_->codec_id);
    if (!codec) {
      return errors::Internal("could not find video codec: ", codec_context_->codec_id);
    }
    // Initialize the decoders
    // TODO (yongtang): avcodec_open2 is not thread-safe
    AVDictionary *opts = NULL;
    if (avcodec_open2(codec_context_, codec, &opts) < 0) {
      return errors::Internal("could not open codec");
    }

    // Allocate frame
    frame_ = av_frame_alloc();
    if (!frame_) {
      return errors::Internal("could not allocate frame");
    }

    // Initialize packet
    av_init_packet(&packet_);
    packet_.data = NULL;
    packet_.size = 0;

    // create scaling context
    sws_context_ = sws_getContext(codec_context_->width, codec_context_->height, codec_context_->pix_fmt, codec_context_->width, codec_context_->height, AV_PIX_FMT_RGB24, 0, NULL, NULL, NULL);
    if (!sws_context_) {
      return errors::Internal("could not allocate sws context");
    }
    frame_rgb_ = av_frame_alloc();
    if (!frame_rgb_) {
      return errors::Internal("could not allocate rgb frame");
    }
    // Determine required buffer size and allocate buffer
    num_bytes_ = av_image_get_buffer_size(AV_PIX_FMT_RGB24, codec_context_->width, codec_context_->height, 1);
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
	av_packet_unref(&packet_);
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
    av_frame_free(&frame_rgb_);
    sws_freeContext(sws_context_);
    av_frame_free(&frame_);
    avformat_close_input(&format_context_);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
