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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DATASET_OP_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

class ATDSDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "ATDSDatum";
  static constexpr const char* const kFileNames = "filenames";
  static constexpr const char* const kBatchSize = "batch_size";
  static constexpr const char* const kDropRemainder = "drop_remainder";
  static constexpr const char* const kReaderBufferSize = "reader_buffer_size";
  static constexpr const char* const kShuffleBufferSize = "shuffle_buffer_size";
  static constexpr const char* const kNumParallelCalls = "num_parallel_calls";
  static constexpr const char* const kFeatureKeys = "feature_keys";
  static constexpr const char* const kFeatureTypes = "feature_types";
  static constexpr const char* const kSparseDtypes = "sparse_dtypes";
  static constexpr const char* const kSparseShapes = "sparse_shapes";
  static constexpr const char* const kOutputDtypes = "output_dtypes";
  static constexpr const char* const kOutputShapes = "output_shapes";

  static constexpr const char* const kDenseType = "dense";
  static constexpr const char* const kSparseType = "sparse";
  static constexpr const char* const kVarlenType = "varlen";

  explicit ATDSDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;

  std::vector<string> feature_keys_, feature_types_;
  std::vector<DataType> sparse_dtypes_, output_dtypes_;
  std::vector<PartialTensorShape> sparse_shapes_, output_shapes_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DATASET_OP_H_
