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

#include "kernels/ffmpeg_reader.h"

namespace tensorflow {
namespace data {
namespace audio {

class AudioReader : public FFmpegReader{
 public:
  explicit AudioReader(SizedRandomAccessInputStreamInterface* s, const string& filename) : FFmpegReader(s, filename) {}

  virtual ~AudioReader() {};

  Status ReadHeader();
  Status ReadSample(int16 *buffer);
  int64 Channels() { return codec_context_->channels; }
 private:
  int DecodeFrame(int *got_frame) override;
  void ProcessFrame() override;
  enum AVMediaType MediaType() override { return AVMEDIA_TYPE_AUDIO; }

  int64 sample_index_ = 0;
};
}  // namespace video
}  // namespace data
}  // namespace tensorflow
