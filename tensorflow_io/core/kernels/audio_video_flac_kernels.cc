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
#include "FLAC/stream_encoder.h"

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
      case DT_UINT8:
        // convert to unsigned by adding 0x80
        for (int64 channel = 0; channel < frame->header.channels; channel++) {
          for (int64 index = 0; index < samples_to_read; index++) {
            int64 sample_index = p->sample_index + index - p->sample_start;
            p->sample_value->tensor<uint8, 2>()(sample_index, channel) =
                (static_cast<uint8>(buffer[channel][index] + 0x80));
          }
        }
        break;
      case DT_INT16:
        for (int64 channel = 0; channel < frame->header.channels; channel++) {
          for (int64 index = 0; index < samples_to_read; index++) {
            int64 sample_index = p->sample_index + index - p->sample_start;
            p->sample_value->tensor<int16, 2>()(sample_index, channel) =
                buffer[channel][index];
          }
        }
        break;
      case DT_INT32:
        // left shift 8 bit as we want to fill int32
        for (int64 channel = 0; channel < frame->header.channels; channel++) {
          for (int64 index = 0; index < samples_to_read; index++) {
            int64 sample_index = p->sample_index + index - p->sample_start;
            p->sample_value->tensor<int32, 2>()(sample_index, channel) =
                (static_cast<int32>(buffer[channel][index]) << 8);
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

class FlacStreamEncoder {
 public:
  FlacStreamEncoder(tstring* buffer) : buffer(buffer), offset(0) {}
  ~FlacStreamEncoder() {}

  static const int64 kSampleBufferCount = 1024;

  static FLAC__StreamEncoderWriteStatus WriteCallback(
      const FLAC__StreamEncoder* encoder, const FLAC__byte buffer[],
      size_t bytes, uint32_t samples, uint32_t current_frame,
      void* client_data) {
    FlacStreamEncoder* p = static_cast<FlacStreamEncoder*>(client_data);
    if (p->offset + bytes > p->buffer->size()) {
      p->buffer->resize(p->offset + bytes);
    }
    memcpy(&(*p->buffer)[p->offset], buffer, bytes);
    p->offset += bytes;
    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
  }

  static FLAC__StreamEncoderSeekStatus SeekCallback(
      const FLAC__StreamEncoder* encoder, FLAC__uint64 absolute_byte_offset,
      void* client_data) {
    FlacStreamEncoder* p = static_cast<FlacStreamEncoder*>(client_data);
    if (absolute_byte_offset > p->buffer->size()) {
      return FLAC__STREAM_ENCODER_SEEK_STATUS_ERROR;
    }
    p->offset = absolute_byte_offset;
    return FLAC__STREAM_ENCODER_SEEK_STATUS_OK;
  }

  static FLAC__StreamEncoderTellStatus TellCallback(
      const FLAC__StreamEncoder* encoder, FLAC__uint64* absolute_byte_offset,
      void* client_data) {
    FlacStreamEncoder* p = static_cast<FlacStreamEncoder*>(client_data);
    *absolute_byte_offset = p->offset;
    return FLAC__STREAM_ENCODER_TELL_STATUS_OK;
  }

  static void MetadataCallback(const FLAC__StreamEncoder* encoder,
                               const FLAC__StreamMetadata* metadata,
                               void* client_data) {}

  tstring* buffer;
  int64 offset;
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
      case 8:
        dtype = DT_UINT8;
        break;
      case 16:
        dtype = DT_INT16;
        break;
      case 24:
        dtype = DT_INT32;
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
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  std::unique_ptr<FLAC__StreamDecoder, void (*)(FLAC__StreamDecoder*)> decoder_;
  std::unique_ptr<FlacStreamDecoder> stream_decoder_;
};

class AudioDecodeFlacOp : public OpKernel {
 public:
  explicit AudioDecodeFlacOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->input("shape", &shape_tensor));

    const tstring& input = input_tensor->scalar<tstring>()();

    std::unique_ptr<FlacReadableResource> resource(
        new FlacReadableResource(env_));
    OP_REQUIRES_OK(context,
                   resource->Init("memory", input.data(), input.size()));

    int32 rate;
    DataType dtype;
    TensorShape shape;
    OP_REQUIRES_OK(context, resource->Spec(&shape, &dtype, &rate));

    OP_REQUIRES(context, (dtype == context->expected_output_dtype(0)),
                errors::InvalidArgument(
                    "dtype mismatch: ", DataTypeString(dtype), " vs. ",
                    DataTypeString(context->expected_output_dtype(0))));

    PartialTensorShape provided_shape;
    OP_REQUIRES_OK(context, PartialTensorShape::MakePartialShape(
                                shape_tensor->flat<int64>().data(),
                                shape_tensor->NumElements(), &provided_shape));
    OP_REQUIRES(context, (provided_shape.IsCompatibleWith(shape)),
                errors::InvalidArgument(
                    "shape mismatch: ", provided_shape.DebugString(), " vs. ",
                    shape.DebugString()));

    OP_REQUIRES_OK(
        context,
        resource->Read(0, shape.dim_size(0),
                       [&](const TensorShape& shape, Tensor** value) -> Status {
                         TF_RETURN_IF_ERROR(
                             context->allocate_output(0, shape, value));
                         return Status::OK();
                       }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class AudioEncodeFlacOp : public OpKernel {
 public:
  explicit AudioEncodeFlacOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* rate_tensor;
    OP_REQUIRES_OK(context, context->input("rate", &rate_tensor));

    const int64 rate = rate_tensor->scalar<int64>()();
    const int64 samples = input_tensor->shape().dim_size(0);
    const int64 channels = input_tensor->shape().dim_size(1);

    int64 bytes_per_sample;
    switch (input_tensor->dtype()) {
      case DT_UINT8:
        bytes_per_sample = 1;
        break;
      case DT_INT16:
        bytes_per_sample = 2;
        break;
      case DT_INT32:
        bytes_per_sample = 3;
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "data type ", DataTypeString(input_tensor->dtype()),
                        " not supported"));
    }

    std::unique_ptr<FLAC__StreamEncoder, void (*)(FLAC__StreamEncoder*)>
        encoder(nullptr, [](FLAC__StreamEncoder* p) {
          if (p != nullptr) {
            FLAC__stream_encoder_delete(p);
          }
        });
    encoder.reset(FLAC__stream_encoder_new());

    FLAC__bool ok;

    ok = FLAC__stream_encoder_set_verify(encoder.get(), true);
    OP_REQUIRES(context, ok, errors::InvalidArgument("unable to set verify"));

    // TODO: compression level could be a input tensor node passed in.
    // ok = FLAC__stream_encoder_set_compression_level(encoder.get(), 5);
    // OP_REQUIRES(context, ok, errors::InvalidArgument("unable to set
    // compression level"));

    ok = FLAC__stream_encoder_set_channels(encoder.get(), channels);
    OP_REQUIRES(context, ok, errors::InvalidArgument("unable to set channels"));

    ok = FLAC__stream_encoder_set_bits_per_sample(encoder.get(),
                                                  bytes_per_sample * 8);
    OP_REQUIRES(context, ok,
                errors::InvalidArgument("unable to set bits per sample"));

    ok = FLAC__stream_encoder_set_sample_rate(encoder.get(), rate);
    OP_REQUIRES(context, ok, errors::InvalidArgument("unable to set rate"));

    ok =
        FLAC__stream_encoder_set_total_samples_estimate(encoder.get(), samples);
    OP_REQUIRES(
        context, ok,
        errors::InvalidArgument("unable to set total samples estimate"));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));

    tstring& output = output_tensor->scalar<tstring>()();

    std::unique_ptr<FlacStreamEncoder> stream_encoder;
    stream_encoder.reset(new FlacStreamEncoder(&output));

    FLAC__StreamEncoderInitStatus s = FLAC__stream_encoder_init_stream(
        encoder.get(), FlacStreamEncoder::WriteCallback,
        FlacStreamEncoder::SeekCallback, FlacStreamEncoder::TellCallback,
        FlacStreamEncoder::MetadataCallback, stream_encoder.get());
    OP_REQUIRES(context, (s == FLAC__STREAM_ENCODER_INIT_STATUS_OK),
                errors::InvalidArgument("unable to initialize stream: ", s));

    std::unique_ptr<FLAC__int32[]> pcm(
        new FLAC__int32[FlacStreamEncoder::kSampleBufferCount * channels]);

    int64 count = 0;
    while (count < samples) {
      int64 chunk = (count + FlacStreamEncoder::kSampleBufferCount < samples)
                        ? (FlacStreamEncoder::kSampleBufferCount)
                        : (samples - count);
      switch (input_tensor->dtype()) {
        case DT_UINT8:
          // convert to signed by sub 0x80
          for (int64 i = 0; i < chunk; i++) {
            for (int64 c = 0; c < channels; c++) {
              pcm.get()[i * channels + c] =
                  static_cast<int32>(
                      input_tensor->flat<uint8>()((count + i) * channels + c)) -
                  0x80;
            }
          }
          break;
        case DT_INT16:
          for (int64 i = 0; i < chunk; i++) {
            for (int64 c = 0; c < channels; c++) {
              pcm.get()[i * channels + c] =
                  input_tensor->flat<int16>()((count + i) * channels + c);
            }
          }
          break;
        case DT_INT32:
          // right shift 8 bit as int32 was filled
          for (int64 i = 0; i < chunk; i++) {
            for (int64 c = 0; c < channels; c++) {
              pcm.get()[i * channels + c] =
                  (input_tensor->flat<int32>()((count + i) * channels + c) >>
                   8);
            }
          }
          break;
      }
      ok = FLAC__stream_encoder_process_interleaved(encoder.get(), pcm.get(),
                                                    chunk);
      OP_REQUIRES(
          context, ok,
          errors::InvalidArgument("unable to process interleaved stream"));
      count += chunk;
    }

    ok = FLAC__stream_encoder_finish(encoder.get());
    OP_REQUIRES(context, ok,
                errors::InvalidArgument("unable to finish stream"));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>AudioDecodeFlac").Device(DEVICE_CPU),
                        AudioDecodeFlacOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioEncodeFlac").Device(DEVICE_CPU),
                        AudioEncodeFlacOp);
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
