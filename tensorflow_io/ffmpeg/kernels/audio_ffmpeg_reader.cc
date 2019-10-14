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
#include "kernels/audio_ffmpeg_reader.h"

namespace tensorflow {
namespace data {
namespace audio {


Status AudioReader::ReadHeader() {

    Status status = InitializeReader();
    if (!status.ok()) {
      return status;
    }

    int data_size = av_get_bytes_per_sample(codec_context_->sample_fmt);
    if (data_size != 2) {
      return errors::InvalidArgument("only int16 data type (data size == 2) supported, received size: ", data_size);
    }

    frame_more_ = true;
    packet_more_ = false;
    buffer_more_ = ReadAhead(true);

    return Status::OK();
}
Status AudioReader::ReadSample(int16 *buffer) {
    while (buffer_more_) {
      if (sample_index_ < frame_->nb_samples) {
        for (int64 channel = 0; channel < codec_context_->channels; channel++) {
          memcpy(&buffer[channel], frame_->data[channel] + (sample_index_ * sizeof(int16)), sizeof(int16));
        }
        sample_index_++;
        return Status::OK();
      }
      buffer_more_ = ReadAhead(true);
      sample_index_ = 0;
    }
    return errors::OutOfRange("EOF");
}

int AudioReader::DecodeFrame(int *got_frame) {
  int decoded = avcodec_decode_audio4(codec_context_, frame_, got_frame, &packet_);
  if (decoded >= 0) {
    decoded = FFMIN(decoded, packet_.size);
    // Only AV_SAMPLE_FMT_S16 is supported at the moment. Other formats have to be converted.
    if (frame_->format != AV_SAMPLE_FMT_S16) {
      return -1;
    }
  }
  return decoded;
}
void AudioReader::ProcessFrame() {
  // TODO: convert sample to int16 format
}

}  // namespace video
}  // namespace data
}  // namespace tensorflow
