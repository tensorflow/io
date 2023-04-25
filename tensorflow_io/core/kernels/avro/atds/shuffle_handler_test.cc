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

#include "tensorflow_io/core/kernels/avro/atds/shuffle_handler.h"

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_io/core/kernels/avro/atds/avro_block_reader.h"

namespace tensorflow {
namespace data {

class ShuffleTest : public ::testing::Test {
 protected:
  ShuffleTest() { shuffle_handler_ = std::make_unique<ShuffleHandler>(&mu_); }

  void SetUp() override {
    for (size_t i = 0; i < 10; i++) {
      int64 rand_mult = static_cast<int64>(rand() % 5 + 5);
      blocks_.emplace_back(std::make_unique<AvroBlock>(AvroBlock{
          rand_mult * 64,    // int64_t object_count;
          0,                 // int64_t num_to_decode;
          0,                 // int64_t num_decoded;
          100000,            // int64_t byte_count;
          0,                 // int64_t counts;
          tstring("haha"),   // tstring content;
          avro::NULL_CODEC,  // avro::Codec codec;
          4888               // size_t read_offset;
      }));
    }
  }
  mutex mu_;
  std::unique_ptr<ShuffleHandler> shuffle_handler_;
  std::vector<std::unique_ptr<AvroBlock>> blocks_;
};

TEST_F(ShuffleTest, NoShuffleTest) {
  size_t batch_size = 1024;
  size_t shuffle_buffer_size = 0;
  shuffle_handler_->SampleBlocks(batch_size, shuffle_buffer_size > 0, blocks_);
  // assert that the sum of all num_to_decode == batch_size
  size_t sum_of_num_to_decode = 0;
  for (size_t i = 0; i < blocks_.size(); i++) {
    sum_of_num_to_decode += blocks_[i]->num_to_decode;
  }
  EXPECT_EQ(sum_of_num_to_decode, batch_size);
}

TEST_F(ShuffleTest, ShuffleBufferTest) {
  size_t batch_size = 1024;
  size_t shuffle_buffer_size = 2048;
  shuffle_handler_->SampleBlocks(batch_size, shuffle_buffer_size > 0, blocks_);
  // assert that the sum of all num_to_decode == batch_size
  size_t sum_of_num_to_decode = 0;
  for (size_t i = 0; i < blocks_.size(); i++) {
    sum_of_num_to_decode += blocks_[i]->num_to_decode;
  }
  EXPECT_EQ(sum_of_num_to_decode, batch_size);
}

TEST_F(ShuffleTest, UniformDistributionTest) {
  const int64 bin_size = 10;
  int64 bins[bin_size] = {0};  // observed frequencies
  int64 error = 50;  // none of the 10 bins will differ from the avg (1000
                     // datapoints) by more than this
  int64 num = 0;
  int64 num_samples = 1000;
  int64 avg = num_samples / bin_size;
  int64 k = 0;
  int64 idx = 0;
  while (k < num_samples) {
    num = shuffle_handler_->Random() % num_samples;
    idx = num /
          avg;  // 0-99 goes to bucket 0, 100-199 goes to bucket 1 and so on.
    bins[idx]++;
    k++;
  }
  // check uniformity by ensuring that every bin is near the avg num of points
  for (int i = 0; i < bin_size; i++) {
    EXPECT_NEAR(bins[i], avg, error);
  }
}

}  // namespace data
}  // namespace tensorflow
