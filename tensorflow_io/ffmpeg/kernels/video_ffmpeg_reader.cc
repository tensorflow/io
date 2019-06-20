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

#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/file_system.h"
#include "kernels/video_ffmpeg_reader.h"

namespace tensorflow {
namespace data {
namespace video {

Status VideoReader::ReadHeader() {

    Status status = InitializeReader();
    if (!status.ok()) {
      return status;
    }

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

int VideoReader::DecodeFrame(int *got_frame) {
  return avcodec_decode_video2(codec_context_, frame_, got_frame, &packet_);
}

void VideoReader::ProcessFrame() {
  sws_scale(sws_context_, frame_->data, frame_->linesize, 0, codec_context_->height, frame_rgb_->data, frame_rgb_->linesize);
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
}

}  // namespace video
}  // namespace data
}  // namespace tensorflow
