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

#include "kernels/ffmpeg_reader.h"

namespace tensorflow {
namespace data {
namespace video {
class VideoReader : public FFmpegReader {
 public:
  explicit VideoReader(SizedRandomAccessInputStreamInterface* s, const string& filename) : FFmpegReader(s, filename) {}

  virtual ~VideoReader();

  Status ReadHeader();
  Status ReadFrame(int *num_bytes, uint8_t**value, int *height, int *width);
 private:
  int DecodeFrame(int *got_frame) override;
  void ProcessFrame() override;

  enum AVMediaType MediaType() override { return AVMEDIA_TYPE_VIDEO; }

  uint8_t *buffer_rgb_ = 0;
  AVFrame *frame_rgb_ = 0;
  struct SwsContext *sws_context_ = 0;
};

}  // namespace video
}  // namespace data
}  // namespace tensorflow
