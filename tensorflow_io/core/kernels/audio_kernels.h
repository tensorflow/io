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

#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {
namespace data {

enum AudioFileFormat {
  UnknownFormat = 0,
  WavFormat = 1,
  FlacFormat = 2,
  OggFormat = 3,
  Mp4Format = 4,
  Mp3Format = 5
};

class AudioReadableResourceBase : public ResourceBase {
 public:
  virtual Status Init(const string& input) = 0;
  virtual Status Read(
      const int64 start, const int64 stop,
      std::function<Status(const TensorShape& shape, Tensor** value)>
          allocate_func) = 0;
  virtual Status Spec(TensorShape* shape, DataType* dtype, int32* rate) = 0;
};

Status WAVReadableResourceInit(
    Env* env, const string& input,
    std::unique_ptr<AudioReadableResourceBase>& resource);
Status OggReadableResourceInit(
    Env* env, const string& input,
    std::unique_ptr<AudioReadableResourceBase>& resource);
Status FlacReadableResourceInit(
    Env* env, const string& input,
    std::unique_ptr<AudioReadableResourceBase>& resource);
Status MP3ReadableResourceInit(
    Env* env, const string& input,
    std::unique_ptr<AudioReadableResourceBase>& resource);
Status MP4ReadableResourceInit(
    Env* env, const string& input,
    std::unique_ptr<AudioReadableResourceBase>& resource);

class DecodedAudio {
public:
  const bool success;
  const int channels;
  const int samples_perchannel;
  const int sampling_rate;
  // should first contain all samples of the left channel
  // followed by the right channel
  const int16 *data;

  size_t data_size();

  DecodedAudio(bool success, size_t channels, size_t samples_perchannel,
               size_t sampling_rate, int16 *data);
  ~DecodedAudio();
};

std::unique_ptr<DecodedAudio> DecodeMP3(StringPiece &data);

}  // namespace data
}  // namespace tensorflow
