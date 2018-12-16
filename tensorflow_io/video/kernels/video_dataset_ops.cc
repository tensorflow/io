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

extern "C" {

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
#include <dlfcn.h>

}

namespace tensorflow {
namespace data {
namespace {

class VideoReader {
 public:
  explicit VideoReader(const string &filename)
      : filename_(filename) {}

  Status ReadHeader() {
    // Open input file, and allocate format context
    if (avformat_open_input(&format_context_, filename_.c_str(), NULL, NULL) < 0) {
      return errors::InvalidArgument("could not open video file: ", filename_);
    }
    // Retrieve stream information
    if (avformat_find_stream_info(format_context_, NULL) < 0) {
      return errors::InvalidArgument("could not find stream information: ", filename_);
    }
    // Find video stream
    if ((stream_index_ = av_find_best_stream(format_context_, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0)) < 0) {
      return errors::InvalidArgument("could not find video stream: ", filename_);
    }

    AVStream *video_stream = format_context_->streams[stream_index_];
    // Find decoder for the stream
    AVCodec *codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
    if (!codec) {
      return errors::Internal("could not find video codec: ", video_stream->codecpar->codec_id);
    }
    // Allocate a codec context for the decoder
    codec_context_ = avcodec_alloc_context3(codec);
    if (!codec_context_) {
      return errors::Internal("could not allocate codec context");
    }
    // Copy codec parameters from input stream to output codec context
    if (avcodec_parameters_to_context(codec_context_, video_stream->codecpar) < 0) {
      return errors::Internal("could not copy codec parameters from input stream to output codec context");
    }
    // Initialize the decoders
    // TODO (yongtang): avcodec_open2 is not thread-safe
    AVDictionary *opts = NULL;
    if (avcodec_open2(codec_context_, codec, &opts) < 0) {
      return errors::Internal("could not open codec");
    }

    // Allocate frame
    frame_ = av_frame_alloc();
    if (!frame_) {
      return errors::Internal("could not allocate frame");
    }

    // Initialize packet
    av_init_packet(&packet_);
    packet_.data = NULL;
    packet_.size = 0;

    // create scaling context
    sws_context_ = sws_getContext(codec_context_->width, codec_context_->height, codec_context_->pix_fmt, codec_context_->width, codec_context_->height, AV_PIX_FMT_RGB24, 0, NULL, NULL, NULL);
    if (!sws_context_) {
      return errors::Internal("could not allocate sws context");
    }
    frame_rgb_ = av_frame_alloc();
    if (!frame_rgb_) {
      return errors::Internal("could not allocate rgb frame");
    }
    // Determine required buffer size and allocate buffer
    num_bytes_ = av_image_get_buffer_size(AV_PIX_FMT_RGB24, codec_context_->width, codec_context_->height, 1);
    buffer_rgb_ = (uint8_t *)av_malloc(num_bytes_ * sizeof(uint8_t));
    avpicture_fill((AVPicture *)frame_rgb_, buffer_rgb_, AV_PIX_FMT_RGB24, codec_context_->width, codec_context_->height);

    frame_more_ = true;
    packet_more_ = false;
    buffer_more_ = ReadAhead(true);

    return Status::OK();
  }

  bool ReadAhead(bool first) {
    while (packet_more_ || frame_more_) {
      while (packet_more_) {
        packet_more_ = false;
	if (packet_.stream_index == stream_index_) {
          int got_frame = 0;
          int decoded = avcodec_decode_video2(codec_context_, frame_, &got_frame, &packet_);
          if (decoded >= 0 && got_frame) {
	    sws_scale(sws_context_, frame_->data, frame_->linesize, 0, codec_context_->height, frame_rgb_->data, frame_rgb_->linesize);
	    if (packet_.data) {
	      packet_.data += decoded;
	      packet_.size -= decoded;
              packet_more_ = (packet_.size > 0);
	    }
            return true;
          }
	}
      }
      if (frame_more_) {
        // If this is not the first time, unref the packet
	av_packet_unref(&packet_);
	frame_more_ = (av_read_frame(format_context_, &packet_) == 0); 
	if (!frame_more_) {
          // Flush out the cached packet
	  packet_more_ = true;
          packet_.data = NULL;
          packet_.size = 0;
	} else {
	  // More packet to process
          packet_more_ = true;
	}
      }
    }
    return false;
  }

  Status ReadFrame(int *num_bytes, uint8_t**value, int *height, int *width) {
    *height = codec_context_->height;
    *width = codec_context_->width;
    *num_bytes = num_bytes_;
    if (buffer_more_) {
      *value = buffer_rgb_;
      buffer_more_ = ReadAhead(true);
      return Status::OK();
    }
    return errors::OutOfRange("EOF");
  }

  virtual ~VideoReader() {
    av_free(buffer_rgb_);
    av_frame_free(&frame_rgb_);
    sws_freeContext(sws_context_);
    av_frame_free(&frame_);
    avcodec_free_context(&codec_context_);
    avformat_close_input(&format_context_);
  }

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
  TF_DISALLOW_COPY_AND_ASSIGN(VideoReader);
};

static mutex mu(LINKER_INITIALIZED);
static unsigned count(0);
void VideoReaderInit() {
  mutex_lock lock(mu);
  count++;
  if (count == 1) {
    // Register all formats and codecs
    av_register_all();
  }
}

class VideoDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit VideoDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
  }
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    *output = new Dataset(ctx, filenames);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::vector<string>& filenames)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(filenames) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Video")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_UINT8});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{-1, -1, 3}});
      return *shapes;
    }

    string DebugString() const override {
      return "VideoDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {filenames}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a file, so try to read the next record.
          if (reader_) {
            int num_bytes, height, width;
	    uint8_t *value;
            Status status = reader_->ReadFrame(&num_bytes, &value, &height, &width);
            if (!errors::IsOutOfRange(status)) {
              TF_RETURN_IF_ERROR(status);

              Tensor value_tensor(ctx->allocator({}), DT_UINT8, {height, width, 3});
	      std::memcpy(reinterpret_cast<char*>(value_tensor.flat<uint8_t>().data()), reinterpret_cast<char*>(value), num_bytes * sizeof(uint8_t));
              out_tensors->emplace_back(std::move(value_tensor));

              *end_of_sequence = false;
              return Status::OK();
            }
            // We have reached the end of the current file, so maybe
            // move on to next file.
            ResetStreamsLocked();
            ++current_file_index_;
          }

          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        } while (true);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal is currently not supported");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "RestoreInternal is currently not supported");
      }

     private:
      // Sets up Video streams to read from the topic at
      // `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
	VideoReaderInit();

        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }

        // Actually move on to next file.
        const string& filename = dataset()->filenames_[current_file_index_];
        reader_.reset(new VideoReader(filename));
        return reader_->ReadHeader();
	return Status::OK();
      }

      // Resets all Video streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        reader_.reset();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<VideoReader> reader_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
  };
};

REGISTER_KERNEL_BUILDER(Name("VideoDataset").Device(DEVICE_CPU),
                        VideoDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
