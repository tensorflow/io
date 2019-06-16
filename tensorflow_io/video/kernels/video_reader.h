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
#include "kernels/dataset_ops.h"

extern "C" {

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
#include <dlfcn.h>

}

namespace tensorflow {
namespace data {
namespace video {

class VideoReader {
 public:
  explicit VideoReader(SizedRandomAccessInputStreamInterface* s, const string& filename) : stream_(s), filename_(filename) {}

  Status ReadHeader();

  bool ReadAhead(bool first);

  Status ReadFrame(int *num_bytes, uint8_t**value, int *height, int *width);

  virtual ~VideoReader();

 public:
  SizedRandomAccessInputStreamInterface* stream_;
  int64 offset_ = 0;
 private:
  std::string ahead_;
  std::string filename_;
  bool frame_more_ = false;
  bool packet_more_ = false;
  bool buffer_more_ = false;
  int stream_index_ = -1;
  size_t num_bytes_ = 0;
  uint8_t *buffer_rgb_ = 0;
  AVFrame *frame_rgb_ = 0;
  struct SwsContext *sws_context_ = 0;
  AVFormatContext *format_context_ = 0;
  AVCodecContext *codec_context_ = 0;
  AVFrame *frame_ = 0;
  AVPacket packet_;
  AVIOContext *io_context_ = NULL;
  TF_DISALLOW_COPY_AND_ASSIGN(VideoReader);
};

}  // namespace
}  // namespace data
}  // namespace tensorflow
