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

#include "tensorflow_io/core/kernels/audio_kernels.h"

#include "speex/speex_resampler.h"

namespace tensorflow {
namespace data {

// DecodedAudio
size_t DecodedAudio::data_size() {
  return channels * samples_perchannel * sizeof(int16);
}

DecodedAudio::DecodedAudio(bool success, size_t channels,
                           size_t samples_perchannel, size_t sampling_rate,
                           int16 *data)
    : success(success), channels(channels),
      samples_perchannel(samples_perchannel), sampling_rate(sampling_rate),
      data(data) {}

DecodedAudio::~DecodedAudio() {
  if (data) {
    std::free((void *)data);
  }
}

// DecodeAudioBaseOp
DecodeAudioBaseOp::DecodeAudioBaseOp(OpKernelConstruction *context) : OpKernel(context) {}

void DecodeAudioBaseOp::Compute(OpKernelContext *context) {
  // get the input data, i.e. encoded audio data
  const Tensor &input_tensor = context->input(0);
  const string &input_data = input_tensor.scalar<tstring>()();
  StringPiece data (input_data.data(), input_data.size());

  // decode audio
  std::unique_ptr<DecodedAudio> decoded = decode(data, nullptr);

  // make sure decoding was successful
  OP_REQUIRES(
      context, decoded->success,
      errors::InvalidArgument("Audio data could not be decoded"));

  // output 1: samples
  Tensor *output_tensor = nullptr;
  TensorShape output_shape {decoded->channels, decoded->samples_perchannel};
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, output_shape, &output_tensor));

  // copy data from decoder buffer into output tensor
  auto output_flat = output_tensor->flat<int16>();
  std::memcpy(output_flat.data(), decoded->data, decoded->data_size());

  // output 2: sample rate
  Tensor *sample_rate_output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                   &sample_rate_output));
  sample_rate_output->flat<int32>()(0) = decoded->sampling_rate;
}

namespace {

AudioFileFormat ClassifyAudioFileFormat(const char *data) {
  // currently requires 8 bytes of data
  if (std::memcmp(data, "RIFF", 4) == 0) {
    return WavFormat;
  } else if (std::memcmp(data, "OggS", 4) == 0) {
    return OggFormat;
  } else if (std::memcmp(data, "fLaC", 4) == 0) {
    return FlacFormat;
  } else if (std::memcmp(data + 4, "ftyp", 4) == 0) {
    return Mp4Format;
  } else if (std::memcmp(data, "ID3", 3) == 0) {
    // TODO MP3 files do not necessarily have to have an ID3 header
    return Mp3Format;
  } else {
    return UnknownFormat;
  }
}

class AudioReadableResource : public AudioReadableResourceBase {
 public:
  AudioReadableResource(Env* env) : env_(env), resource_(nullptr) {}
  ~AudioReadableResource() {}

  Status Init(const string& input) override {
    mutex_lock l(mu_);
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(input, &file));
    char header[8];
    StringPiece result;
    TF_RETURN_IF_ERROR(file->Read(0, sizeof(header), &result, header));
    switch (ClassifyAudioFileFormat(header)) {
    case WavFormat:
      return WAVReadableResourceInit(env_, input, resource_);
    case OggFormat:
      return OggReadableResourceInit(env_, input, resource_);
    case FlacFormat:
      return FlacReadableResourceInit(env_, input, resource_);
    case Mp4Format:
      LOG(ERROR) << "MP4A file is not fully supported!";
      return MP4ReadableResourceInit(env_, input, resource_);
    default:
      // currently we are trying MP3 as a default option
      Status status = MP3ReadableResourceInit(env_, input, resource_);
      if (status.ok()) {
        return status;
      }
    }
    return errors::InvalidArgument("unknown file type: ", input);
  }
  Status Spec(TensorShape* shape, DataType* dtype, int32* rate) override {
    mutex_lock l(mu_);
    return resource_->Spec(shape, dtype, rate);
  }
  Status Read(const int64 start, const int64 stop,
              std::function<Status(const TensorShape& shape, Tensor** value)>
                  allocate_func) override {
    mutex_lock l(mu_);
    return resource_->Read(start, stop, allocate_func);
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return resource_->DebugString();
  }

 protected:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<AudioReadableResourceBase> resource_ GUARDED_BY(mu_);
};

class AudioReadableInitOp : public ResourceOpKernel<AudioReadableResource> {
 public:
  explicit AudioReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<AudioReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<AudioReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    OP_REQUIRES_OK(context, resource_->Init(input_tensor->scalar<string>()()));
  }
  Status CreateResource(AudioReadableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new AudioReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class AudioReadableSpecOp : public OpKernel {
 public:
  explicit AudioReadableSpecOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    AudioReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    TensorShape shape;
    DataType dtype;
    int32 rate;
    OP_REQUIRES_OK(context, resource->Spec(&shape, &dtype, &rate));

    Tensor* shape_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({2}), &shape_tensor));
    shape_tensor->flat<int64>()(0) = shape.dim_size(0);
    shape_tensor->flat<int64>()(1) = shape.dim_size(1);

    Tensor* dtype_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &dtype_tensor));
    dtype_tensor->scalar<int64>()() = dtype;

    Tensor* rate_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({}), &rate_tensor));
    rate_tensor->scalar<int32>()() = rate;
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class AudioReadableReadOp : public OpKernel {
 public:
  explicit AudioReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    AudioReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    int64 stop = stop_tensor->scalar<int64>()();

    OP_REQUIRES_OK(
        context,
        resource->Read(start, stop,
                       [&](const TensorShape& shape, Tensor** value) -> Status {
                         TF_RETURN_IF_ERROR(
                             context->allocate_output(0, shape, value));
                         return Status::OK();
                       }));
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class AudioResampleOp : public OpKernel {
 public:
  explicit AudioResampleOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("quality", &quality_));
    OP_REQUIRES(
        context,
        (SPEEX_RESAMPLER_QUALITY_MIN <= quality_ &&
         quality_ <= SPEEX_RESAMPLER_QUALITY_MAX),
        errors::InvalidArgument("quality ", quality_, " not supported, need [",
                                SPEEX_RESAMPLER_QUALITY_MIN, ", ",
                                SPEEX_RESAMPLER_QUALITY_MAX, "]"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* rate_in_tensor;
    OP_REQUIRES_OK(context, context->input("rate_in", &rate_in_tensor));
    const int64 rate_in = rate_in_tensor->scalar<int64>()();

    const Tensor* rate_out_tensor;
    OP_REQUIRES_OK(context, context->input("rate_out", &rate_out_tensor));
    const int64 rate_out = rate_out_tensor->scalar<int64>()();

    int64 samples_in = input_tensor->shape().dim_size(0);
    int64 channels = input_tensor->shape().dim_size(1);

    std::unique_ptr<SpeexResamplerState, void (*)(SpeexResamplerState*)> state(
        nullptr, [](SpeexResamplerState* p) {
          if (p != nullptr) {
            speex_resampler_destroy(p);
          }
        });

    int err = 0;
    state.reset(
        speex_resampler_init(channels, rate_in, rate_out, quality_, &err));
    OP_REQUIRES(
        context, (state.get() != nullptr),
        errors::InvalidArgument("unable to initialize resampler: ", err));

    int64 samples_out = samples_in * rate_out / rate_in;
    Tensor* output_tensor;
    switch (input_tensor->dtype()) {
      case DT_INT16: {
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape({samples_out, channels}),
                                    &output_tensor));

        uint32_t processed_in = samples_in;
        uint32_t processed_out = samples_out;
        int returned = speex_resampler_process_interleaved_int(
            state.get(), input_tensor->flat<int16>().data(), &processed_in,
            output_tensor->flat<int16>().data(), &processed_out);
        OP_REQUIRES(context, (returned == 0),
                    errors::InvalidArgument("process error: ", returned));
        OP_REQUIRES(
            context, (processed_out == samples_out),
            errors::InvalidArgument("output buffer mismatch: ", processed_out,
                                    " vs. ", samples_out));
      } break;
      case DT_FLOAT: {
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape({samples_out, channels}),
                                    &output_tensor));
        uint32_t processed_in = samples_in;
        uint32_t processed_out = samples_out;
        int returned = speex_resampler_process_interleaved_float(
            state.get(), input_tensor->flat<float>().data(), &processed_in,
            output_tensor->flat<float>().data(), &processed_out);
        OP_REQUIRES(context, (returned == 0),
                    errors::InvalidArgument("process error: ", returned));
        OP_REQUIRES(
            context, (processed_out == samples_out),
            errors::InvalidArgument("output buffer mismatch: ", processed_out,
                                    " vs. ", samples_out));
      } break;
      default:
        OP_REQUIRES_OK(context,
                       errors::InvalidArgument(
                           "Data type ", DataTypeString(input_tensor->dtype()),
                           " not supported"));
    }
  }

 private:
  int64 quality_;
};

class DecodeAudioOp : public DecodeAudioBaseOp {
 public:
  DecodeAudioOp(OpKernelConstruction *context) : DecodeAudioBaseOp(context) {}

  std::unique_ptr<DecodedAudio> decode(StringPiece &data, void *config) {
    auto error = std::unique_ptr<DecodedAudio>(new DecodedAudio(false, 0, 0, 0, nullptr));
    switch (ClassifyAudioFileFormat(data.data())) {
      case WavFormat:
        LOG(ERROR) << "Direct decoding of WAV not yet supported.";
        return error;
      case OggFormat:
        LOG(ERROR) << "Direct decoding of Ogg not yet supported.";
        return error;
      case FlacFormat:
        LOG(ERROR) << "Direct decoding of Flac not yet supported.";
        return error;
      case Mp4Format:
        LOG(ERROR) << "Direct decoding of Mp4 not yet supported.";
        return error;
      default:
        // currently we are trying MP3 as a default option
        return DecodeMP3(data);
   }
  }
};

REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableInit").Device(DEVICE_CPU),
                        AudioReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableSpec").Device(DEVICE_CPU),
                        AudioReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableRead").Device(DEVICE_CPU),
                        AudioReadableReadOp);

REGISTER_KERNEL_BUILDER(Name("IO>AudioResample").Device(DEVICE_CPU),
                        AudioResampleOp);

REGISTER_KERNEL_BUILDER(Name("IO>AudioDecode").Device(DEVICE_CPU),
                        DecodeAudioOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
