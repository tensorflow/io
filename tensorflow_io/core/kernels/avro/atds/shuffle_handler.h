/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SHUFFLE_HANDLER_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SHUFFLE_HANDLER_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow_io/core/kernels/avro/atds/avro_block_reader.h"

namespace tensorflow {
namespace data {

class ShuffleHandler {
 public:
  ShuffleHandler(mutex* mu) {
    mu_ = mu;
    ResetRngs();
  }
  void SampleBlocks(size_t batch_size, bool shuffle,
                    std::vector<std::unique_ptr<AvroBlock>>& blocks) {
    size_t i = 0;
    size_t block_size = blocks.size();
    // LOG(INFO) << "shuffle batch size " << batch_size << " shuffle block size:
    // " << block_size;
    if (!shuffle) {
      size_t j = 0;
      while (i < batch_size) {
        auto& random_block = blocks[j];
        random_block->num_to_decode =
            std::min(random_block->object_count - random_block->num_decoded,
                     static_cast<int64_t>(batch_size - i));
        i += random_block->num_to_decode;
        if ((random_block->num_decoded + random_block->num_to_decode) ==
            random_block->object_count) {
          j++;
        }
      }
    } else {
      while (i < batch_size) {
        size_t block_id = Random() % block_size;
        // LOG(INFO) << "shuffle block size " << block_size << " block_id: " <<
        // block_id << " actual block size: " << blocks.size();
        auto& random_block = blocks[block_id];
        int64 remaining = random_block->object_count -
                          random_block->num_decoded -
                          random_block->num_to_decode;
        if (remaining > 0) {
          // Decode the whole block when it has less than 1/10 of the undecoded
          // records. It is to quickly recycle the almost decoded blocks.
          int64 decode_all_threshold = random_block->object_count / 10;
          size_t decode_num = 1;
          if (remaining <= decode_all_threshold) {
            decode_num =
                std::min(static_cast<size_t>(remaining), batch_size - i);
          }
          random_block->num_to_decode += decode_num;
          i += decode_num;
        }
      }
    }
    // update counts so that the elements don't have huge gaps
    for (size_t k = 0; k < block_size; k++) {
      blocks[k]->counts = blocks[k]->num_to_decode;
      if (k > 0) {
        blocks[k]->counts += blocks[k - 1]->counts;
      }
      // LOG(INFO) << "block " << k << " object count: " <<
      // blocks[k]->object_count << " counts: " << blocks[k]->counts
      //            << " num_decoded: " << blocks[k]->num_decoded << "
      //            num_to_decode: " << blocks[k]->num_to_decode;
    }
  }
  // function to produce random numbers
  random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random()
      TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
    num_random_samples_++;
    return generator_->operator()();
  }

  void ResetRngs() TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
    // Reset the generators based on the current iterator seeds.
    int64 seed_ = random::New64();
    int64 seed2_ = random::New64();
    parent_generator_ = std::make_unique<random::PhiloxRandom>(seed_, seed2_);
    generator_ =
        std::make_unique<random::SingleSampleAdapter<random::PhiloxRandom>>(
            parent_generator_.get());
    generator_->Skip(num_random_samples_);
    num_random_samples_ = 0;
  }

 private:
  // this is not owned by ShuffleHandler. This is owned by the calling class
  mutex* mu_;
  int64 num_random_samples_ TF_GUARDED_BY(*mu_) = 0;
  std::unique_ptr<random::PhiloxRandom> parent_generator_ TF_GUARDED_BY(*mu_);
  std::unique_ptr<random::SingleSampleAdapter<random::PhiloxRandom>> generator_
      TF_GUARDED_BY(*mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SHUFFLE_HANDLER_H_
