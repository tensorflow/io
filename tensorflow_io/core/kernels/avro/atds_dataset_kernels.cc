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

#include "tensorflow_io/core/kernels/avro/atds_dataset_kernels.h"

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <cstring>
#include <vector>

#include "api/Compiler.hh"
#include "api/DataFile.hh"
#include "api/Decoder.hh"
#include "api/Specific.hh"
#include "api/Stream.hh"
#include "api/ValidSchema.hh"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow_io/core/kernels/avro/atds/atds_decoder.h"
#include "tensorflow_io/core/kernels/avro/atds/avro_block_reader.h"
#include "tensorflow_io/core/kernels/avro/atds/decompression_handler.h"
#include "tensorflow_io/core/kernels/avro/atds/errors.h"
#include "tensorflow_io/core/kernels/avro/atds/shuffle_handler.h"

namespace tensorflow {
namespace data {

void ParallelFor(const std::function<void(size_t)>& f, size_t n,
                 thread::ThreadPool* thread_pool) {
  if (n == 0) return;
  if (thread_pool == nullptr) {
    for (size_t i = 0; i < n; ++i) {
      f(i);
    }
  } else {
    BlockingCounter counter(n - 1);
    for (size_t i = 1; i < n; ++i) {
      thread_pool->Schedule([i, &f, &counter] {
        f(i);
        counter.DecrementCount();
      });
    }
    f(0);
    counter.Wait();
  }
}

/* static */ constexpr const char* const ATDSDatasetOp::kDatasetType;
/* static */ constexpr const char* const ATDSDatasetOp::kFileNames;
/* static */ constexpr const char* const ATDSDatasetOp::kBatchSize;
/* static */ constexpr const char* const ATDSDatasetOp::kDropRemainder;
/* static */ constexpr const char* const ATDSDatasetOp::kReaderBufferSize;
/* static */ constexpr const char* const ATDSDatasetOp::kShuffleBufferSize;
/* static */ constexpr const char* const ATDSDatasetOp::kNumParallelCalls;
/* static */ constexpr const char* const ATDSDatasetOp::kFeatureKeys;
/* static */ constexpr const char* const ATDSDatasetOp::kFeatureTypes;
/* static */ constexpr const char* const ATDSDatasetOp::kSparseDtypes;
/* static */ constexpr const char* const ATDSDatasetOp::kSparseShapes;
/* static */ constexpr const char* const ATDSDatasetOp::kOutputDtypes;
/* static */ constexpr const char* const ATDSDatasetOp::kOutputShapes;
/* static */ constexpr const char* const ATDSDatasetOp::kDenseType;
/* static */ constexpr const char* const ATDSDatasetOp::kSparseType;
/* static */ constexpr const char* const ATDSDatasetOp::kVarlenType;

class ATDSDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, std::vector<tstring> filenames,
                   size_t batch_size, bool drop_remainder,
                   int64 reader_buffer_size, int64 shuffle_buffer_size,
                   int64 num_parallel_calls,
                   const std::vector<string>& feature_keys,
                   const std::vector<string>& feature_types,
                   const std::vector<DataType>& sparse_dtypes,
                   const std::vector<PartialTensorShape>& sparse_shapes,
                   const std::vector<DataType>& output_dtypes,
                   const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        filenames_(std::move(filenames)),
        batch_size_(batch_size),
        reader_buffer_size_(reader_buffer_size),
        shuffle_buffer_size_(shuffle_buffer_size),
        num_parallel_calls_(num_parallel_calls),
        drop_remainder_(drop_remainder),
        feature_keys_(feature_keys),
        feature_types_(feature_types),
        sparse_dtypes_(sparse_dtypes),
        sparse_shapes_(sparse_shapes),
        output_dtypes_(output_dtypes),
        output_shapes_(output_shapes) {
    size_t num_of_features = feature_keys_.size();
    output_tensor_types_.reserve(num_of_features);
    sparse_value_index_.reserve(sparse_dtypes.size());
    for (size_t i = 0; i < num_of_features; i++) {
      if (feature_types[i] == kDenseType) {
        output_tensor_types_.emplace_back(TensorType::dense);
        auto dim_v = output_shapes[i].dim_sizes();
        size_t rank = dim_v.size();

        TensorShapeProto proto;
        PartialTensorShape shape;
        for (size_t d = 1; d < rank; d++) {
          proto.add_dim()->set_size(dim_v[d]);
        }
        if (!PartialTensorShape::BuildPartialTensorShape(proto, &shape).ok()) {
          LOG(ERROR) << "Error encountered in creating PartialTensorShape for "
                        "dense features.";
        }
        dense_features_.emplace_back(atds::FeatureType::dense, feature_keys_[i],
                                     output_dtypes[i], shape, num_of_dense_);
        num_of_dense_++;
      } else if (feature_types[i] == kSparseType ||
                 feature_types[i] == kVarlenType) {
        output_tensor_types_.emplace_back(TensorType::sparse);

        auto& shape = sparse_shapes[num_of_sparse_];
        // The estimated number of elements in this sparse tensor.
        // The estimated number is used to preallocate sparse value buffer.
        size_t estimated_elements = 1;
        if (feature_types[i] == kVarlenType) {
          for (auto dim : shape) {
            // Assume unknown dim will only have 1 element. For example,
            // varlen tensor with shape [-1, 2, -1] is expected to have 2
            // elements in total.
            if (dim.size > 0) {
              estimated_elements *= dim.size;
            }
          }
        }
        size_t rank_after_batch = static_cast<size_t>(shape.dims() + 1);
        sparse_expected_elements_.indices.push_back(rank_after_batch *
                                                    estimated_elements);

        size_t values_index = 0;
        auto dtype = sparse_dtypes[num_of_sparse_];
        if (dtype == DT_INT32) {
          values_index = sparse_dtype_counts_.int_counts++;
          sparse_expected_elements_.int_values.push_back(estimated_elements);
        } else if (dtype == DT_INT64) {
          values_index = sparse_dtype_counts_.long_counts++;
          sparse_expected_elements_.long_values.push_back(estimated_elements);
        } else if (dtype == DT_FLOAT) {
          values_index = sparse_dtype_counts_.float_counts++;
          sparse_expected_elements_.float_values.push_back(estimated_elements);
        } else if (dtype == DT_DOUBLE) {
          values_index = sparse_dtype_counts_.double_counts++;
          sparse_expected_elements_.double_values.push_back(estimated_elements);
        } else if (dtype == DT_STRING) {
          values_index = sparse_dtype_counts_.string_counts++;
          sparse_expected_elements_.string_values.push_back(estimated_elements);
        } else if (dtype == DT_BOOL) {
          values_index = sparse_dtype_counts_.bool_counts++;
          sparse_expected_elements_.bool_values.push_back(estimated_elements);
        }
        sparse_value_index_.emplace_back(values_index);

        if (feature_types[i] == kSparseType) {
          sparse_features_.emplace_back(
              atds::FeatureType::sparse, feature_keys_[i],
              sparse_dtypes[num_of_sparse_], sparse_shapes[num_of_sparse_],
              num_of_sparse_, values_index);
        } else if (feature_types[i] == kVarlenType) {
          varlen_features_.emplace_back(
              atds::FeatureType::varlen, feature_keys_[i],
              sparse_dtypes[num_of_sparse_], sparse_shapes[num_of_sparse_],
              num_of_sparse_, values_index);
        }
        num_of_sparse_++;
      } else {
        LOG(ERROR) << "Unknown feature type " << feature_types[i];
      }
    }

    for (auto& dtype : output_dtypes) {
      output_dtype_vector_.push_back(dtype);
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    return output_dtype_vector_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return OkStatus();
  }

  Status CheckExternalState() const override { return OkStatus(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* filenames = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
    Node* batch_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
    Node* drop_remainder = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder));
    Node* reader_buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(reader_buffer_size_, &reader_buffer_size));
    Node* shuffle_buffer_size = nullptr;
    TF_RETURN_IF_ERROR(
        b->AddScalar(shuffle_buffer_size_, &shuffle_buffer_size));
    Node* num_parallel_calls = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(num_parallel_calls_, &num_parallel_calls));

    AttrValue feature_keys;
    b->BuildAttrValue(feature_keys_, &feature_keys);
    AttrValue feature_types;
    b->BuildAttrValue(feature_types_, &feature_types);
    AttrValue sparse_dtypes;
    b->BuildAttrValue(sparse_dtypes_, &sparse_dtypes);
    AttrValue sparse_shapes;
    b->BuildAttrValue(sparse_shapes_, &sparse_shapes);
    AttrValue output_dtypes;
    b->BuildAttrValue(output_dtypes_, &output_dtypes);
    AttrValue output_shapes;
    b->BuildAttrValue(output_shapes_, &output_shapes);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {filenames, batch_size, drop_remainder, reader_buffer_size,
         shuffle_buffer_size, num_parallel_calls},
        {{kFeatureKeys, feature_keys},
         {kFeatureTypes, feature_types},
         {kSparseDtypes, sparse_dtypes},
         {kSparseShapes, sparse_shapes},
         {kOutputDtypes, output_dtypes},
         {kOutputShapes, output_shapes}},
        output));
    return OkStatus();
  }

 private:
  enum class TensorType { dense, sparse };

  /**
   * Utility struct to collect the number of sparse tensors for each DType.
   */
  struct SparseDtypeCounts {
    size_t int_counts = 0;
    size_t long_counts = 0;
    size_t float_counts = 0;
    size_t double_counts = 0;
    size_t string_counts = 0;
    size_t bool_counts = 0;
  };

  /**
   * Utility struct to store the estimated number of elements for each sparse
   * tensor. The estimated number in values tensor and indices tensor are
   * ordered based on the layout in atds::sparse::ValueBuffer.
   * The information is used for better buffer pre-allocation.
   */
  struct SparseExpectedElements {
    std::vector<size_t> int_values;
    std::vector<size_t> long_values;
    std::vector<size_t> float_values;
    std::vector<size_t> double_values;
    std::vector<size_t> string_values;
    std::vector<size_t> bool_values;
    std::vector<size_t> indices;
  };

  class Iterator : public DatasetIterator<Dataset> {
   public:
    static constexpr const char* const kWaitingForData = "WaitingForData";
    static constexpr const char* const kBlockReading = "BlockReading";
    static constexpr const char* const kParsingThread = "ParsingThread_";
    static constexpr const char* const kDeflateDecompression =
        "DeflateDecompression";
    static constexpr const char* const kSnappyDecompression =
        "SnappyDecompression";
    static constexpr const char* const kFillingSparseValues =
        "FillingSparseValues";

    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          shuffle_handler_(nullptr),
          cond_var_(std::make_shared<condition_variable>()),
          write_var_(std::make_shared<condition_variable>()),
          mu_(std::make_shared<mutex>()),
          count_(0) {
      batch_size_ = static_cast<size_t>(dataset()->batch_size_);
      shuffle_buffer_size_ =
          static_cast<size_t>(dataset()->shuffle_buffer_size_);
      shuffle_handler_ = std::make_unique<ShuffleHandler>(mu_.get());
      decompression_handler_ = std::make_unique<DecompressionHandler>();
      auto& sparse_dtype_counts = dataset()->sparse_dtype_counts_;
      value_buffer_.int_values.resize(sparse_dtype_counts.int_counts);
      value_buffer_.long_values.resize(sparse_dtype_counts.long_counts);
      value_buffer_.float_values.resize(sparse_dtype_counts.float_counts);
      value_buffer_.double_values.resize(sparse_dtype_counts.double_counts);
      value_buffer_.string_values.resize(sparse_dtype_counts.string_counts);
      value_buffer_.bool_values.resize(sparse_dtype_counts.bool_counts);
      value_buffer_.num_of_elements.resize(dataset()->num_of_sparse_);
      value_buffer_.indices.resize(dataset()->num_of_sparse_);
    }

    ~Iterator() override {
      // must ensure that the thread is cancelled.
      CancelThreads();
      // LOG(INFO) << "Decompression time per record (us): "  <<
      // (static_cast<double>(GetTotalStats(total_decompress_micros_)) /
      // GetTotalStats(num_decompressed_objects_)); LOG(INFO) << "Decode time
      // per record (us): " <<
      // (static_cast<double>(GetTotalStats(total_decode_micros_)) /
      // GetTotalStats(total_records_parsed_));
    }

    void CancelThreads() TF_LOCKS_EXCLUDED(mu_) {
      mutex_lock l(*mu_);
      mutex_lock i(input_mu_);
      cancelled_ = true;
      cond_var_->notify_all();
      write_var_->notify_all();
      // wait for thread to finish
      if (prefetch_thread_) {
        while (!prefetch_thread_finished_) {
          write_var_->wait(i);
        }
      }
    }

    Status Initialize(IteratorContext* ctx) {
      int64 num_threads = dataset()->num_parallel_calls_;
      const int64 max_parallelism = port::MaxParallelism();
      if (num_threads <= 0 || num_threads > max_parallelism) {
        if (num_threads == tensorflow::data::model::kAutotune) {
          LOG(INFO) << "Thread autotuning enabled for "
                       "ATDSDatasetOp::Dataset::Iterator.";
        }
        LOG(INFO) << "Create ATDSDatasetOp::Dataset::Iterator thread pool with "
                  << "the maximum parallelism number " << max_parallelism
                  << " for this process.";
        num_threads = max_parallelism;
      }
      thread_delays.resize(max_parallelism, 0);
      thread_itrs.resize(max_parallelism, 0);
      thread_pool_ =
          ctx->CreateThreadPool(std::string(kDatasetType), num_threads);
      return OkStatus();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(*mu_);
      EnsurePrefetchThreadStarted(ctx);
      size_t total_buffer = total_buffer_size();
      while (true) {
        // LOG(INFO) << "b " << blocks_.size() << " c_: " << count_;
        // while count_ is smaller than batch_size, wait on cond_var_ if not
        // last file this will get woken up by the prefetch thread
        size_t count = 0;
        bool prefetch_thread_finished = false;
        {
          tensorflow::profiler::TraceMe trace(kWaitingForData);

          mutex_lock i(input_mu_);
          while (!cancelled_ && !prefetch_thread_finished_ &&
                 count_ < total_buffer) {
            // LOG(INFO) << "waiting on block refill " << blocks_.size() << "
            // count: " << count_;
            write_var_->notify_all();
            cond_var_->wait(i);
          }
          // LOG(INFO) << "done waiting on block refill " << blocks_.size() << "
          // count: " << count_;
          if (cancelled_) {
            return OkStatus();
          }

          count_ = 0;
          // merge write_blocks_ into blocks_
          blocks_.reserve(blocks_.size() + write_blocks_.size());
          blocks_.insert(blocks_.end(),
                         std::make_move_iterator(write_blocks_.begin()),
                         std::make_move_iterator(write_blocks_.end()));
          write_blocks_.clear();  // size down the write_blocks

          size_t non_empty_idx = 0;
          for (size_t i = 0; i < blocks_.size(); i++) {
            count_ += blocks_[i]->object_count - blocks_[i]->num_decoded;
            if (blocks_[i]->num_decoded < blocks_[i]->object_count) {
              std::swap(blocks_[non_empty_idx], blocks_[i]);
              non_empty_idx++;
            }
          }
          blocks_.resize(non_empty_idx);

          count = count_;
          prefetch_thread_finished = prefetch_thread_finished_;

          // let it continue to read batch_size_ or count_ records.
          count_ -= std::min(count_, batch_size_);
          write_var_->notify_all();

          if (prefetch_thread_finished_) {
            // Finished epoch, reset shuffle for new epoch
            shuffle_handler_->ResetRngs();
          }
        }

        bool drop_remainder = dataset()->drop_remainder_;
        if (count >= batch_size_ ||
            (!drop_remainder && prefetch_thread_finished && count > 0)) {
          // LOG(INFO) << "Process "  <<  blocks_.size() << " blocks with " <<
          // count << " objects. " << non_empty_idx << " batch: " <<
          // batch_size_;
          size_t batch_size = std::min(count, batch_size_);
          PartialTensorShape batch_dim({static_cast<int64>(batch_size)});
          auto num_of_dense = dataset()->num_of_dense_;
          auto num_of_sparse = dataset()->num_of_sparse_;
          auto& dense_features = dataset()->dense_features_;
          std::vector<Tensor> dense_tensors;
          for (size_t i = 0; i < num_of_dense; i++) {
            auto& dense_feature = dense_features[i];
            TensorShape shape;
            batch_dim.Concatenate(dense_feature.shape).AsTensorShape(&shape);
            dense_tensors.emplace_back(ctx->allocator({}), dense_feature.dtype,
                                       shape);
          }

          size_t thread_pool_size =
              static_cast<size_t>(thread_pool_->NumThreads());
          size_t num_blocks = blocks_.size();
          size_t num_threads = std::min(num_blocks, thread_pool_size);
          num_threads = std::min(num_threads,
                                 static_cast<size_t>(port::MaxParallelism()));

          int64 user_defined_thread_num = dataset()->num_parallel_calls_;
          if (user_defined_thread_num > 0) {
            num_threads = std::min(
                num_threads, static_cast<size_t>(user_defined_thread_num));
          } else if (user_defined_thread_num ==
                     tensorflow::data::model::kAutotune) {
            num_threads = ComputeNumAutotuneThreads(num_threads);
          }
          total_records_parsed_.resize(num_threads, 0);
          total_decode_micros_.resize(num_threads, 0);
          num_decompressed_objects_.resize(num_threads, 0);
          total_decompress_micros_.resize(num_threads, 0);
          shuffle_handler_->SampleBlocks(batch_size, shuffle_buffer_size_ > 0,
                                         blocks_);
          std::vector<atds::sparse::ValueBuffer> sparse_buffer(num_threads,
                                                               value_buffer_);

          std::vector<Status> status_of_threads(num_threads);
          auto process_block = [&](size_t i, size_t thread_idx,
                                   avro::DecoderPtr& decoder,
                                   atds::sparse::ValueBuffer& buffer,
                                   std::vector<avro::GenericDatum>& skipped) {
            // start is the offset in the each example, and therefore just need
            // to be different from every other block.
            size_t start = 0;
            if (i > 0) {
              start += blocks_[i - 1]->counts;
            }
            size_t end = blocks_[i]->counts;
            // LOG(INFO) << "Block: " << i << " start: " << start << " end: " <<
            // end << " read_so_far " << blocks_[i]->num_decoded
            //   << " num_to_decode: " << blocks_[i]->num_to_decode << "
            //   remaining: " << (blocks_[i]->object_count -
            //   blocks_[i]->num_decoded);
            avro::Codec codec = blocks_[i]->codec;
            avro::InputStreamPtr input_stream = nullptr;
            uint64 decompress_start_time = ctx->env()->NowMicros();
            if (codec == avro::NULL_CODEC) {
              input_stream =
                  decompression_handler_->decompressNullCodec(*(blocks_[i]));
            } else if (codec == avro::DEFLATE_CODEC) {
              tensorflow::profiler::TraceMe traceme(kDeflateDecompression);
              input_stream =
                  decompression_handler_->decompressDeflateCodec(*(blocks_[i]));
            }
#ifdef SNAPPY_CODEC_AVAILABLE
            else if (codec == avro::SNAPPY_CODEC) {
              tensorflow::profiler::TraceMe traceme(kSnappyDecompression);
              input_stream =
                  decompression_handler_->decompressSnappyCodec(*(blocks_[i]));
            }
#endif
            else {
              throw avro::Exception(
                  "Unsupported Avro codec. Only null or deflate is supported. "
                  "Got " +
                  codec);
            }
            uint64 decompress_end_time = ctx->env()->NowMicros();
            if (codec != avro::NULL_CODEC) {
              total_decompress_micros_[thread_idx] +=
                  (decompress_end_time - decompress_start_time);
              num_decompressed_objects_[thread_idx] += blocks_[i]->object_count;
              // LOG(INFO) << "Block " << i << " decompress time (us): " <<
              // (decompress_end_time - decompress_start_time)
              //     << ", num records: " << blocks_[i]->object_count;
            }
            decoder->init(*input_stream);

            while (start < end) {
              // LOG(INFO) << "Block: " << i << " start: " << start;
              uint64 datum_parse_start = ctx->env()->NowMicros();
              auto decoding_status = atds_decoder_->DecodeATDSDatum(
                  decoder, dense_tensors, buffer, skipped, start);
              if (!decoding_status.ok()) {
                // The decoding of this block has failed,
                // setting the number of decoded objects to the total number of
                // objects in the block so the decoder will skip decoding this
                // block.
                blocks_[i]->num_decoded = blocks_[i]->object_count;
                return decoding_status;
              }
              uint64 datum_parse_end = ctx->env()->NowMicros();
              total_decode_micros_[thread_idx] +=
                  (datum_parse_end - datum_parse_start);
              total_records_parsed_[thread_idx] += 1;
              start++;
              blocks_[i]->num_decoded++;
              blocks_[i]->num_to_decode--;
            }

            if (blocks_[i]->object_count > blocks_[i]->num_decoded) {
              decoder->init(*input_stream);
              blocks_[i]->read_offset += input_stream->byteCount();
              // LOG(INFO) << "Block: " << i << " Reset offset to " <<
              // blocks_[i]->read_offset << ". " << (end - start)
              //           << " datum left for block " << i;
            }
            // LOG(INFO) << "process block " << i << " . Read: " <<
            // blocks_[i]->num_decoded;
            return OkStatus();
          };

          std::vector<size_t> block_nums;
          GetBlockRanges(num_threads, block_nums);
          std::vector<uint64> thread_start_times;
          thread_start_times.resize(num_threads, 0);
          auto process = [&](size_t index) {
            auto parsing_thread_name = [index]() {
              return strings::StrCat(kParsingThread, index);
            };
            tensorflow::profiler::TraceMe trace(parsing_thread_name);

            thread_start_times[index] = ctx->env()->NowMicros();
            size_t block_start = 0;
            if (index > 0) {
              block_start = block_nums[index - 1];
            }
            size_t block_end = block_nums[index];
            auto decoder = avro::binaryDecoder();
            auto skipped = atds_decoder_->GetSkippedData();
            auto& buffer = sparse_buffer[index];
            size_t count_start = 0;
            if (block_start > 0) {
              count_start = blocks_[block_start - 1]->counts;
            }
            size_t num_of_datum = blocks_[block_end - 1]->counts - count_start;
            InitSparseValueBuffer(buffer, num_of_datum);
            // LOG(INFO) << "Thread " << index << " process blocks from " <<
            // block_start << " to "
            //           << block_end << " with " << num_of_datum << "
            //           examples.";

            status_of_threads[index] = OkStatus();
            auto& status = status_of_threads[index];

            for (size_t i = block_start; i < block_end && status.ok(); i++) {
              if (blocks_[i]->codec != avro::NULL_CODEC ||
                  blocks_[i]->num_to_decode > 0) {
                status = process_block(i, index, decoder, buffer, skipped);
              }
            }
            // LOG(INFO) << "Thread " << index << " process blocks from " <<
            // block_start << " to " << block_end << ". Done.";
          };
          ParallelFor(process, num_threads, thread_pool_.get());
          uint64 earliest_start_time = *std::min_element(
              thread_start_times.begin(), thread_start_times.end());
          for (size_t i = 0; i < num_threads; i++) {
            thread_delays[i] += (thread_start_times[i] - earliest_start_time);
            thread_itrs[i] += 1;
          }
          for (Status& status : status_of_threads) {
            TF_RETURN_IF_ERROR(status);
          }

          std::vector<int64> num_of_elements(num_of_sparse, 0);
          std::vector<Tensor> indices_tensors;
          std::vector<Tensor> values_tensors;
          std::vector<Tensor> shape_tensors;
          indices_tensors.reserve(num_of_sparse);
          values_tensors.reserve(num_of_sparse);
          shape_tensors.reserve(num_of_sparse);
          auto& sparse_dtypes = dataset()->sparse_dtypes_;
          auto& sparse_shapes = dataset()->sparse_shapes_;
          for (size_t i = 0; i < num_of_sparse; i++) {
            for (size_t t = 0; t < num_threads; t++) {
              // Check if vector is empty and move on to the next vector.
              // If shuffle buffer and number of threads is large compared
              // to the batch, this vector maybe empty for certain threads.
              num_of_elements[i] += static_cast<int64>(
                  GetLastElement(sparse_buffer[t].num_of_elements[i]));
            }
            auto& sparse_shape = sparse_shapes[i];

            int64 rank = sparse_shape.dims() + 1;
            TensorShape indices_shape({num_of_elements[i], rank});
            TensorShape values_shape({num_of_elements[i]});
            TensorShape shape_shape({rank});
            indices_tensors.emplace_back(DT_INT64, indices_shape);
            values_tensors.emplace_back(sparse_dtypes[i], values_shape);
            shape_tensors.emplace_back(DT_INT64, shape_shape);

            auto& shape_tensor = shape_tensors.back();
            size_t d = 0;
            shape_tensor.vec<long>()(d++) = batch_size;
            for (auto dim : sparse_shape) {
              if (dim.size > 0) {
                shape_tensor.vec<long>()(d++) = dim.size;
              } else {
                // When dim size is unknown i.e. -1, scan indices array to find
                // the largest dim value.
                long max_dim = -1;
                for (size_t t = 0; t < num_threads; t++) {
                  auto& indices = sparse_buffer[t].indices[i];
                  for (size_t pos = d; pos < indices.size(); pos += rank) {
                    max_dim = std::max(max_dim, indices[pos]);
                  }
                }
                shape_tensor.vec<long>()(d++) = max_dim + 1;
              }
            }
          }

          auto& sparse_value_index = dataset()->sparse_value_index_;
          auto fill_sparse_value = [&](int64 thread_index) {
            // LOG(INFO) << "Thread " << thread_index << " starts filling sparse
            // value";
            auto& buffer = sparse_buffer[thread_index];
            for (size_t i = 0; i < num_of_sparse; i++) {
              size_t offset = 0;
              int64 index = thread_index;
              while (index > 0) {
                index--;
                offset +=
                    GetLastElement(sparse_buffer[index].num_of_elements[i]);
              }

              size_t rank_after_batch =
                  static_cast<size_t>(sparse_shapes[i].dims() + 1);
              atds::sparse::FillIndicesTensor(buffer.indices[i],
                                              indices_tensors[i],
                                              rank_after_batch * offset);
              atds::sparse::FillValuesTensor(buffer, values_tensors[i],
                                             sparse_dtypes[i],
                                             sparse_value_index[i], offset);
              // LOG(INFO) << "Thread " << thread_index << " filled sparse
              // values.";
            }
          };

          {
            tensorflow::profiler::TraceMe trace(kFillingSparseValues);
            ParallelFor(fill_sparse_value, num_threads, thread_pool_.get());
          }

          size_t feature_num = num_of_dense + num_of_sparse;
          size_t dense_index = 0, sparse_index = 0;
          auto& feature_types = dataset()->output_tensor_types_;
          for (size_t i = 0; i < feature_num; i++) {
            if (feature_types[i] == TensorType::dense) {
              out_tensors->emplace_back(
                  std::move(dense_tensors[dense_index++]));
            } else if (feature_types[i] == TensorType::sparse) {
              out_tensors->emplace_back(DT_VARIANT, TensorShape({3}));
              auto& serialized_sparse_t = out_tensors->back();
              serialized_sparse_t.vec<Variant>()(0) =
                  std::move(indices_tensors[sparse_index]);
              serialized_sparse_t.vec<Variant>()(1) =
                  std::move(values_tensors[sparse_index]);
              serialized_sparse_t.vec<Variant>()(2) =
                  std::move(shape_tensors[sparse_index]);
              sparse_index++;
            }
          }
          // LOG(INFO) << "Done with batch " ;
          *end_of_sequence = false;
          return OkStatus();
        } else {
          *end_of_sequence = true;
          return prefetch_thread_status_;
        }
      }
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return errors::Unimplemented("SaveInternal");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented(
          "Iterator does not support 'RestoreInternal')");
    }

   private:
    // Returns the last element of the provided integer vector is a null-safe
    // fashion
    size_t GetLastElement(const std::vector<size_t>& num_of_elements_at_i) {
      if (num_of_elements_at_i.empty()) {
        return 0;
      }
      return num_of_elements_at_i.back();
    }

    void PrefetchThread(const std::shared_ptr<IteratorContext>& ctx) {
      size_t total_buffer = total_buffer_size();
      std::unique_ptr<AvroBlockReader> reader;
      std::unique_ptr<tensorflow::RandomAccessFile> file;
      size_t current_file_index = 0;
      while (true) {
        // 1. wait for a slot in the buffer
        {
          mutex_lock l(input_mu_);
          while (!cancelled_ && count_ >= total_buffer) {
            // LOG(INFO) << "prefetch waiting on block size " << blocks_.size()
            // << " count: " << count_;
            cond_var_->notify_one();
            write_var_->wait(l);
          }
          // LOG(INFO) << "prefetch done waiting on block size " <<
          // blocks_.size() << " count: " << count_;
          if (cancelled_) {
            prefetch_thread_finished_ = true;
            prefetch_thread_status_ = OkStatus();
            cond_var_->notify_all();
            write_var_->notify_all();
            return;
          }
        }  // done with mutex_lock l
        // 2. read the next elements unil count hits max
        Status status = OkStatus();
        if (!reader) {
          status =
              SetupStreamsLocked(ctx->env(), file, reader, current_file_index);
          if (!status.ok()) {
            mutex_lock l(input_mu_);
            LOG(ERROR) << "Error loading file: "
                       << dataset()->filenames_[current_file_index];
            prefetch_thread_finished_ = true;
            prefetch_thread_status_ = status;
            cond_var_->notify_all();
            write_var_->notify_all();
            return;
          }
        }

        // LOG(INFO) << "Before processing " << count_ << " datum left in
        // block.";
        tensorflow::profiler::TraceMe trace(kBlockReading);

        auto block = std::make_unique<AvroBlock>();
        status = reader->ReadBlock(*block);
        // LOG(INFO) << "Read block status: " << status.ToString();
        // done with mutex_lock input_l
        if (!status.ok()) {
          if (!errors::IsOutOfRange(status)) {
            LOG(ERROR) << "Error in reading avro block. Cause: "
                       << status.ToString();
          }
          // LOG(INFO) << "Resetting stream: " << status.ToString() << "b " <<
          // blocks_.size() << " c_: " << count_;
          ResetStreamsLocked(file, reader);
          ++current_file_index;
          if (current_file_index >= dataset()->filenames_.size()) {
            mutex_lock l(input_mu_);
            prefetch_thread_finished_ = true;
            // Note: this is overwriting any previous errors
            prefetch_thread_status_ = OkStatus();
            cond_var_->notify_all();
            write_var_->notify_all();
            return;
          }  // done with mutex_lock l
        } else {
          mutex_lock n(input_mu_);
          count_ += block->object_count;
          write_blocks_.emplace_back(std::move(block));
          ++num_blocks_read_;
        }
      }  // end while
    }

    Status EnsurePrefetchThreadStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!prefetch_thread_) {
        std::shared_ptr<IteratorContext> new_ctx =
            std::make_shared<IteratorContext>(*ctx);
        prefetch_thread_ =
            ctx->StartThread("atds_data_prefetch",
                             [this, new_ctx]() { PrefetchThread(new_ctx); });
      }
      return OkStatus();
    }

    size_t total_buffer_size() { return batch_size_ + shuffle_buffer_size_; }

    // Sets up reader streams to read from the file at `current_file_index_`.
    Status SetupStreamsLocked(
        Env* env, std::unique_ptr<tensorflow::RandomAccessFile>& file,
        std::unique_ptr<AvroBlockReader>& reader, size_t current_file_index) {
      if (current_file_index >= dataset()->filenames_.size()) {
        return errors::InvalidArgument(
            "current_file_index_:", current_file_index,
            " >= filenames_.size():", dataset()->filenames_.size());
      }

      // Actually move on to next file.
      const string& next_filename = dataset()->filenames_[current_file_index];
      TF_RETURN_IF_ERROR(env->NewRandomAccessFile(next_filename, &file));
      reader = absl::make_unique<AvroBlockReader>(
          file.get(), dataset()->reader_buffer_size_);
      if (atds_decoder_ == nullptr) {
        atds_decoder_ = std::make_unique<atds::ATDSDecoder>(
            dataset()->dense_features_, dataset()->sparse_features_,
            dataset()->varlen_features_);
        TF_RETURN_IF_ERROR(atds_decoder_->Initialize(reader->GetSchema()));
        expected_schema_ = atds_decoder_->GetSchema().toJson(false);
      } else if (expected_schema_ != reader->GetSchema().toJson(false)) {
        string expected_schema = atds_decoder_->GetSchema().toJson(true);
        string varied_schema = reader->GetSchema().toJson(true);
        string filename = dataset()->filenames_[0];
        return atds::VariedSchemaNotSupportedError(
            expected_schema, filename, varied_schema, next_filename);
      }
      return OkStatus();
    }

    // Resets all reader streams.
    void ResetStreamsLocked(std::unique_ptr<tensorflow::RandomAccessFile>& file,
                            std::unique_ptr<AvroBlockReader>& reader) {
      reader.reset();
      file.reset();
    }

    void InitSparseValueBuffer(atds::sparse::ValueBuffer& buffer,
                               size_t num_of_datum) {
      auto& sparse_dtype_counts = dataset()->sparse_dtype_counts_;
      auto& sparse_expected_elements = dataset()->sparse_expected_elements_;
      for (size_t i = 0; i < sparse_dtype_counts.int_counts; i++) {
        buffer.int_values[i].reserve(num_of_datum *
                                     sparse_expected_elements.int_values[i]);
      }
      for (size_t i = 0; i < sparse_dtype_counts.long_counts; i++) {
        buffer.long_values[i].reserve(num_of_datum *
                                      sparse_expected_elements.long_values[i]);
      }
      for (size_t i = 0; i < sparse_dtype_counts.float_counts; i++) {
        buffer.float_values[i].reserve(
            num_of_datum * sparse_expected_elements.float_values[i]);
      }
      for (size_t i = 0; i < sparse_dtype_counts.double_counts; i++) {
        buffer.double_values[i].reserve(
            num_of_datum * sparse_expected_elements.double_values[i]);
      }
      for (size_t i = 0; i < sparse_dtype_counts.string_counts; i++) {
        buffer.string_values[i].reserve(
            num_of_datum * sparse_expected_elements.string_values[i]);
      }
      for (size_t i = 0; i < sparse_dtype_counts.bool_counts; i++) {
        buffer.bool_values[i].reserve(num_of_datum *
                                      sparse_expected_elements.bool_values[i]);
      }

      size_t num_of_sparse = dataset()->num_of_sparse_;
      for (size_t i = 0; i < num_of_sparse; i++) {
        buffer.num_of_elements[i].reserve(num_of_datum);
        buffer.indices[i].reserve(num_of_datum *
                                  sparse_expected_elements.indices[i]);
      }
    }

    void GetUniformBlockRanges(size_t num_threads,
                               std::vector<size_t>& block_nums) {
      size_t num_blocks = blocks_.size();
      size_t blocks_per_thread = num_blocks / num_threads;
      size_t remainder = num_blocks % num_threads;
      size_t block_idx = 0;
      for (size_t i = 0; i < num_threads; i++) {
        block_idx += blocks_per_thread;
        if (i < remainder) {
          block_idx += 1;
        }
        block_nums.emplace_back(block_idx);
      }
    }

    double GetTotalCost(double& decode_cost_per_record,
                        double& decompress_cost_per_record) {
      decode_cost_per_record =
          static_cast<double>(GetTotalStats(total_decode_micros_)) /
          GetTotalStats(total_records_parsed_);
      decompress_cost_per_record = 0;
      double total_cost = decode_cost_per_record * batch_size_;
      if (GetTotalStats(num_decompressed_objects_) > 0) {
        decompress_cost_per_record =
            static_cast<double>(GetTotalStats(total_decompress_micros_)) /
            GetTotalStats(num_decompressed_objects_);
        // Newly read blocks are appended to the end of blocks_ array, and all
        // non-newly read blocks were already decompressed in previous
        // GetNextInternal iterations. So we loop through blocks in reverse
        // order, and terminate when we encounter an already decompressed block
        // (null codec).
        for (size_t i = blocks_.size();
             i > 0 && blocks_[i - 1]->codec != avro::NULL_CODEC; i--) {
          total_cost +=
              (decompress_cost_per_record * blocks_[i - 1]->object_count);
        }
      }
      return total_cost;
    }

    void GetCostBasedBlockRanges(size_t num_threads,
                                 std::vector<size_t>& block_nums) {
      size_t num_blocks = blocks_.size();
      double decode_cost_per_record;
      double decompress_cost_per_record;
      double total_cost =
          GetTotalCost(decode_cost_per_record, decompress_cost_per_record);
      double cost_per_thread = total_cost / num_threads;
      size_t block_idx = 0;
      size_t thread_idx = 0;
      double running_cost = 0;
      while (thread_idx < num_threads) {
        while (running_cost < cost_per_thread * (thread_idx + 1) &&
               block_idx < num_blocks) {
          if (blocks_[block_idx]->codec != avro::NULL_CODEC) {
            running_cost +=
                decompress_cost_per_record * blocks_[block_idx]->object_count;
          }
          running_cost +=
              decode_cost_per_record * blocks_[block_idx]->num_to_decode;
          block_idx++;
        }
        block_nums.emplace_back(block_idx);
        thread_idx++;
      }
      block_nums[num_threads - 1] = num_blocks;
    }

    void GetBlockRanges(size_t num_threads, std::vector<size_t>& block_nums) {
      block_nums.reserve(num_threads);
      if (GetTotalStats(total_decode_micros_) == 0) {
        // No decode time statistics yet. Divide blocks evenly between threads
        GetUniformBlockRanges(num_threads, block_nums);
      } else {
        // Get block ranges per thread based on runtime data
        GetCostBasedBlockRanges(num_threads, block_nums);
      }
    }

    size_t ComputeNumAutotuneThreads(size_t curr_threads) {
      size_t ideal_num_threads = curr_threads;
      if (thread_itrs[0] > 0) {
        double decode_cost_per_record;
        double decompress_cost_per_record;
        double total_cost =
            GetTotalCost(decode_cost_per_record, decompress_cost_per_record);
        double min_cost = std::numeric_limits<double>::max();
        for (size_t i = 1; i < curr_threads; i++) {
          // Compute cost when using `i` threads
          double cost_per_thread = total_cost / i;
          double max_thread_delay = 0;
          for (size_t j = 0; j < i; j++) {
            double thread_delay = 0;
            if (thread_itrs[j] > 0) {
              thread_delay = thread_delays[j] / thread_itrs[j];
            }
            max_thread_delay = std::max(thread_delay, max_thread_delay);
          }
          if (cost_per_thread + max_thread_delay < min_cost) {
            min_cost = cost_per_thread + max_thread_delay;
            ideal_num_threads = i;
          }
        }
      }
      return ideal_num_threads;
    }

    uint64 GetTotalStats(std::vector<uint64>& vec) {
      return std::accumulate(vec.begin(), vec.end(), 0);
    }

    std::unique_ptr<ShuffleHandler> shuffle_handler_ = nullptr;
    std::unique_ptr<DecompressionHandler> decompression_handler_ = nullptr;
    const std::shared_ptr<condition_variable> cond_var_ = nullptr;
    const std::shared_ptr<condition_variable> write_var_ = nullptr;
    size_t batch_size_;
    size_t shuffle_buffer_size_;

    atds::sparse::ValueBuffer value_buffer_;
    std::unique_ptr<thread::ThreadPool> thread_pool_ = nullptr;

    const std::shared_ptr<mutex> mu_;
    std::unique_ptr<Thread> prefetch_thread_ TF_GUARDED_BY(*mu_);
    std::vector<std::unique_ptr<AvroBlock> > blocks_ TF_GUARDED_BY(*mu_);

    mutex input_mu_ TF_ACQUIRED_BEFORE(*mu_);
    size_t count_ TF_GUARDED_BY(input_mu_) = 0;
    bool cancelled_ TF_GUARDED_BY(input_mu_) = false;
    bool prefetch_thread_finished_ TF_GUARDED_BY(input_mu_) = false;
    Status prefetch_thread_status_ TF_GUARDED_BY(input_mu_);
    uint64 num_blocks_read_ TF_GUARDED_BY(input_mu_) = 0;
    std::vector<std::unique_ptr<AvroBlock> > write_blocks_
        TF_GUARDED_BY(input_mu_);

    std::unique_ptr<atds::ATDSDecoder> atds_decoder_ = nullptr;
    string expected_schema_ = "";
    std::vector<uint64> total_records_parsed_ TF_GUARDED_BY(*mu_);
    std::vector<uint64> total_decode_micros_ TF_GUARDED_BY(*mu_);
    std::vector<uint64> num_decompressed_objects_ TF_GUARDED_BY(*mu_);
    std::vector<uint64> total_decompress_micros_ TF_GUARDED_BY(*mu_);
    std::vector<uint64> thread_delays TF_GUARDED_BY(*mu_);
    std::vector<uint64> thread_itrs TF_GUARDED_BY(*mu_);
  };

  const std::vector<tstring> filenames_;
  const int64 batch_size_, reader_buffer_size_, shuffle_buffer_size_,
      num_parallel_calls_;
  const bool drop_remainder_;
  const std::vector<string> feature_keys_, feature_types_;
  const std::vector<DataType> sparse_dtypes_;
  const std::vector<PartialTensorShape> sparse_shapes_;
  const std::vector<DataType> output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
  std::vector<size_t> sparse_value_index_;
  DataTypeVector output_dtype_vector_;

  std::vector<TensorType> output_tensor_types_;

  std::vector<atds::dense::Metadata> dense_features_;
  std::vector<atds::sparse::Metadata> sparse_features_;
  std::vector<atds::varlen::Metadata> varlen_features_;
  SparseDtypeCounts sparse_dtype_counts_;
  SparseExpectedElements sparse_expected_elements_;
  size_t num_of_dense_ = 0, num_of_sparse_ = 0;
};

ATDSDatasetOp::ATDSDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kFeatureKeys, &feature_keys_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kFeatureTypes, &feature_types_));

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kSparseDtypes, &sparse_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kSparseShapes, &sparse_shapes_));

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputDtypes, &output_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));

  auto feature_num = feature_keys_.size();
  OP_REQUIRES(ctx, feature_num == feature_types_.size(),
              errors::InvalidArgument(strings::StrCat(
                  "The length of feature_keys must equal to the ",
                  "length of feature_types. [", feature_num,
                  " != ", feature_types_.size(), "]")));

  OP_REQUIRES(ctx, feature_num == output_dtypes_.size(),
              errors::InvalidArgument(strings::StrCat(
                  "The length of feature_keys must equal to the ",
                  "length of output_dtypes. [", feature_num,
                  " != ", output_dtypes_.size(), "]")));

  OP_REQUIRES(ctx, feature_num == output_shapes_.size(),
              errors::InvalidArgument(strings::StrCat(
                  "The length of feature_keys must equal to the ",
                  "length of output_shapes. [", feature_num,
                  " != ", output_shapes_.size(), "]")));

  size_t num_sparse = 0;
  for (auto& type : feature_types_) {
    OP_REQUIRES(
        ctx, type == kDenseType || type == kSparseType || type == kVarlenType,
        errors::InvalidArgument(strings::StrCat(
            "Invalid feature_type, '", type, "'. Only ", kDenseType, ", ",
            kSparseType, ", and ", kVarlenType, " are supported.")));
    if (type == kSparseType || type == kVarlenType) {
      num_sparse++;
    }
  }

  OP_REQUIRES(ctx, sparse_dtypes_.size() == num_sparse,
              errors::InvalidArgument(strings::StrCat(
                  "The length of sparse_dtypes must equal to the number of ",
                  "sparse features configured in feature_types. [",
                  sparse_dtypes_.size(), " != ", num_sparse, "]")));

  OP_REQUIRES(ctx, sparse_shapes_.size() == num_sparse,
              errors::InvalidArgument(strings::StrCat(
                  "The length of sparse_shapes must equal to the number of ",
                  "sparse features configured in feature_types. [",
                  sparse_shapes_.size(), " != ", num_sparse, "]")));
}

void ATDSDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  const Tensor* filenames_tensor;
  OP_REQUIRES_OK(ctx, ctx->input(kFileNames, &filenames_tensor));
  OP_REQUIRES(
      ctx, filenames_tensor->dims() <= 1,
      errors::InvalidArgument("`filenames` must be a scalar or a vector."));

  std::vector<tstring> filenames;
  filenames.reserve(filenames_tensor->NumElements());
  for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
    VLOG(2) << "Reading file: " << filenames_tensor->flat<tstring>()(i);
    filenames.push_back(filenames_tensor->flat<tstring>()(i));
  }

  int64 batch_size = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kBatchSize, &batch_size));
  OP_REQUIRES(
      ctx, batch_size > 0,
      errors::InvalidArgument(strings::StrCat(
          "`batch_size` must be greater than 0 but found ", batch_size)));

  bool drop_remainder = false;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument<bool>(ctx, kDropRemainder, &drop_remainder));

  int64 reader_buffer_size = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kReaderBufferSize,
                                                 &reader_buffer_size));
  OP_REQUIRES(ctx, reader_buffer_size > 0,
              errors::InvalidArgument(strings::StrCat(
                  "`reader_buffer_size` must be greater than 0 but found ",
                  reader_buffer_size)));

  int64 shuffle_buffer_size = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kShuffleBufferSize,
                                                 &shuffle_buffer_size));
  OP_REQUIRES(
      ctx, shuffle_buffer_size >= 0,
      errors::InvalidArgument(strings::StrCat(
          "`shuffle_buffer_size` must be greater than or equal to 0 but found ",
          shuffle_buffer_size)));

  int64 num_parallel_calls = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kNumParallelCalls,
                                                 &num_parallel_calls));
  OP_REQUIRES(ctx,
              num_parallel_calls > 0 ||
                  num_parallel_calls == tensorflow::data::model::kAutotune,
              errors::InvalidArgument(
                  strings::StrCat("`num_parallel_calls` must be a positive "
                                  "integer or tf.data.AUTOTUNE, got ",
                                  num_parallel_calls)));
  *output = new Dataset(
      ctx, std::move(filenames), batch_size, drop_remainder, reader_buffer_size,
      shuffle_buffer_size, num_parallel_calls, feature_keys_, feature_types_,
      sparse_dtypes_, sparse_shapes_, output_dtypes_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("IO>ATDSDataset").Device(DEVICE_CPU),
                        ATDSDatasetOp);
}  // namespace

}  // namespace data
}  // namespace tensorflow
