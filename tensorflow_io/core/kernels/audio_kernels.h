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

Status PartitionsLookup(const std::vector<int64>& partitions, const int64 start,
                        const int64 stop, int64* upper, int64* lower,
                        int64* extra);

class AudioReadableResourceBase : public ResourceBase {
 public:
  virtual Status Init(const string& filename,
                      const void* optional_memory = nullptr,
                      size_t optional_length = 0) = 0;
  virtual Status Read(
      const int64 start, const int64 stop,
      std::function<Status(const TensorShape& shape, Tensor** value)>
          allocate_func) = 0;
  virtual Status Spec(TensorShape* shape, DataType* dtype, int32* rate) = 0;
};

Status WAVReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource);
Status FlacReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource);
Status OggVorbisReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource);
Status MP3ReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource);
Status MP4AACReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource);

}  // namespace data
}  // namespace tensorflow
