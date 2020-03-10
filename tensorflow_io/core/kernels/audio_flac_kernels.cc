/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_io/core/kernels/audio_kernels.h"

#include "FLAC/stream_decoder.h"

namespace tensorflow {
namespace data {
namespace {

class FlacStreamDecoder {
 public:
  FlacStreamDecoder(SizedRandomAccessFile* file, int64 size)
      : file(file),
        size(size),
        offset(0),
        samples(0),
        channels(0),
        rate(0),
        bits_per_sample(0) {}
  ~FlacStreamDecoder() {}

  void SetTensor(int64 start, Tensor* value) {
    sample_index = start;
    sample_start = start;
    sample_value = value;
  }

  static FLAC__StreamDecoderReadStatus ReadCallback(
      const FLAC__StreamDecoder* decoder, FLAC__byte buffer[], size_t* bytes,
      void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    if (*bytes > 0) {
      if (p->offset >= p->size) {
        *bytes = 0;
        return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
      }
      size_t bytes_to_read =
          (p->offset + *bytes) < (p->size) ? (*bytes) : (p->size - p->offset);
      StringPiece result;
      Status status =
          p->file->Read(p->offset, bytes_to_read, &result, (char*)buffer);
      if (result.size() != bytes_to_read) {
        return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
      }
      *bytes = bytes_to_read;
      p->offset += bytes_to_read;
      return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
    }
    return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
  }
  static FLAC__StreamDecoderSeekStatus SeekCallback(
      const FLAC__StreamDecoder* decoder, FLAC__uint64 absolute_byte_offset,
      void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    if (absolute_byte_offset > p->size) {
      return FLAC__STREAM_DECODER_SEEK_STATUS_ERROR;
    }
    p->offset = absolute_byte_offset;
    return FLAC__STREAM_DECODER_SEEK_STATUS_OK;
  }
  static FLAC__StreamDecoderTellStatus TellCallback(
      const FLAC__StreamDecoder* decoder, FLAC__uint64* absolute_byte_offset,
      void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    *absolute_byte_offset = p->offset;
    return FLAC__STREAM_DECODER_TELL_STATUS_OK;
  }
  static FLAC__StreamDecoderLengthStatus LengthCallback(
      const FLAC__StreamDecoder* decoder, FLAC__uint64* stream_length,
      void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    *stream_length = p->size;
    return FLAC__STREAM_DECODER_LENGTH_STATUS_OK;
  }

  static FLAC__bool EofCallback(const FLAC__StreamDecoder* decoder,
                                void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    return p->offset >= p->size;
  }
  static FLAC__StreamDecoderWriteStatus WriteCallback(
      const FLAC__StreamDecoder* decoder, const FLAC__Frame* frame,
      const FLAC__int32* const buffer[], void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    if (frame->header.channels != p->channels) {
      return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }
    if (frame->header.number_type != FLAC__FRAME_NUMBER_TYPE_SAMPLE_NUMBER) {
      return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }

    if (p->sample_index != frame->header.number.sample_number) {
      return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }

    int64 samples_to_read =
        (p->sample_index + frame->header.blocksize) <
                (p->sample_start + p->sample_value->shape().dim_size(0))
            ? (frame->header.blocksize)
            : (p->sample_start + p->sample_value->shape().dim_size(0) -
               p->sample_index);

    switch (p->sample_value->dtype()) {
      case DT_INT16:
        for (int64 channel = 0; channel < frame->header.channels; channel++) {
          for (int64 index = 0; index < samples_to_read; index++) {
            int64 sample_index = p->sample_index + index - p->sample_start;
            p->sample_value->tensor<int16, 2>()(sample_index, channel) =
                buffer[channel][index];
          }
        }
        break;
      default:
        return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }
    p->sample_index += samples_to_read;
    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
  }

  static void MetadataCallback(const FLAC__StreamDecoder* decoder,
                               const FLAC__StreamMetadata* metadata,
                               void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    p->samples = metadata->data.stream_info.total_samples;
    p->channels = metadata->data.stream_info.channels;
    p->rate = metadata->data.stream_info.sample_rate;
    p->bits_per_sample = metadata->data.stream_info.bits_per_sample;
  }

  static void ErrorCallback(const FLAC__StreamDecoder* decoder,
                            FLAC__StreamDecoderErrorStatus status,
                            void* client_data) {
    std::cerr << "ErrorCallback: " << std::endl;
  }

  SizedRandomAccessFile* file;
  int64 size;
  int64 offset;

  int64 samples;
  int64 channels;
  int64 rate;
  int64 bits_per_sample;

  int64 sample_index;
  int64 sample_start;
  Tensor* sample_value;
};

class FlacReadableResource : public AudioReadableResourceBase {
 public:
  FlacReadableResource(Env* env)
      : env_(env), decoder_(nullptr, [](FLAC__StreamDecoder* p) {
          if (p != nullptr) {
            FLAC__stream_decoder_delete(p);
          }
        }) {}
  ~FlacReadableResource() {}

  Status Init(const string& filename, const void* optional_memory,
              const size_t optional_length) override {
    mutex_lock l(mu_);
    file_.reset(new SizedRandomAccessFile(env_, filename, optional_memory,
                                          optional_length));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    decoder_.reset(FLAC__stream_decoder_new());
    stream_decoder_.reset(new FlacStreamDecoder(file_.get(), file_size_));

    FLAC__StreamDecoderInitStatus s = FLAC__stream_decoder_init_stream(
        decoder_.get(), FlacStreamDecoder::ReadCallback,
        FlacStreamDecoder::SeekCallback, FlacStreamDecoder::TellCallback,
        FlacStreamDecoder::LengthCallback, FlacStreamDecoder::EofCallback,
        FlacStreamDecoder::WriteCallback, FlacStreamDecoder::MetadataCallback,
        FlacStreamDecoder::ErrorCallback, stream_decoder_.get());
    if (s != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
      return errors::InvalidArgument("unable to initialize stream: ", s);
    }
    if (!FLAC__stream_decoder_process_until_end_of_metadata(decoder_.get())) {
      return errors::InvalidArgument("unable to read metadata");
    }

    int64 samples = stream_decoder_->samples;
    int64 channels = stream_decoder_->channels;
    int64 rate = stream_decoder_->rate;
    DataType dtype = DT_INVALID;
    switch (stream_decoder_->bits_per_sample) {
      case 16:
        dtype = DT_INT16;
        break;
      default:
        return errors::InvalidArgument("invalid_bits_per_sample: ",
                                       stream_decoder_->bits_per_sample);
    }

    shape_ = TensorShape({samples, channels});
    dtype_ = dtype;
    rate_ = rate;
    return Status::OK();
  }

  Status Spec(TensorShape* shape, DataType* dtype, int32* rate) override {
    mutex_lock l(mu_);
    *shape = shape_;
    *dtype = dtype_;
    *rate = rate_;
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop,
              std::function<Status(const TensorShape& shape, Tensor** value)>
                  allocate_func) override {
    mutex_lock l(mu_);
    int64 sample_stop =
        (stop < 0) ? (shape_.dim_size(0))
                   : (stop < shape_.dim_size(0) ? stop : shape_.dim_size(0));
    int64 sample_start = (start >= sample_stop) ? sample_stop : start;

    Tensor* value;
    TF_RETURN_IF_ERROR(allocate_func(
        TensorShape({sample_stop - sample_start, shape_.dim_size(1)}), &value));

    stream_decoder_->SetTensor(sample_start, value);
    if (!FLAC__stream_decoder_seek_absolute(decoder_.get(), sample_start)) {
      return errors::InvalidArgument("unable to seek to: ", sample_start);
    }

    while (stream_decoder_->sample_index < sample_stop) {
      if (!FLAC__stream_decoder_process_single(decoder_.get())) {
        return errors::InvalidArgument("unable to read at: ",
                                       stream_decoder_->sample_index);
      }
    }
    return Status::OK();
  }
  string DebugString() const override { return "FlacReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  std::unique_ptr<FLAC__StreamDecoder, void (*)(FLAC__StreamDecoder*)> decoder_;
  std::unique_ptr<FlacStreamDecoder> stream_decoder_;
};

}  // namespace

Status FlacReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource) {
  resource.reset(new FlacReadableResource(env));
  Status status = resource->Init(filename, optional_memory, optional_length);
  if (!status.ok()) {
    resource.reset(nullptr);
  }
  return status;
}

}  // namespace data
}  // namespace tensorflow
