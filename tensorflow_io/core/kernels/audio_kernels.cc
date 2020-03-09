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
namespace {

class AudioReadableResource : public AudioReadableResourceBase {
 public:
  AudioReadableResource(Env* env) : env_(env), resource_(nullptr) {}
  ~AudioReadableResource() {}

  Status Init(const string& filename, const void* optional_memory,
              const size_t optional_length) override {
    mutex_lock l(mu_);
    std::unique_ptr<SizedRandomAccessFile> file;
    file.reset(new SizedRandomAccessFile(env_, filename, optional_memory,
                                         optional_length));
    // Note: check file size is the indicator that the file is valid, as file
    // could be filename only or filename + buffer(memory/length)
    uint64 file_size;
    TF_RETURN_IF_ERROR(file->GetFileSize(&file_size));
    char header[8];
    StringPiece result;
    TF_RETURN_IF_ERROR(file->Read(0, sizeof(header), &result, header));
    if (memcmp(header, "RIFF", 4) == 0) {
      return WAVReadableResourceInit(env_, filename, optional_memory,
                                     optional_length, resource_);
    } else if (memcmp(header, "OggS", 4) == 0) {
      return OggReadableResourceInit(env_, filename, optional_memory,
                                     optional_length, resource_);
    } else if (memcmp(header, "fLaC", 4) == 0) {
      return FlacReadableResourceInit(env_, filename, optional_memory,
                                      optional_length, resource_);
    }
    Status status = MP3ReadableResourceInit(env_, filename, optional_memory,
                                            optional_length, resource_);
    if (status.ok()) {
      return status;
    }
    if (memcmp(&header[4], "ftyp", 4) == 0) {
      LOG(ERROR) << "MP4A file is not fully supported!";
      return MP4ReadableResourceInit(env_, filename, optional_memory,
                                     optional_length, resource_);
    }
    return errors::InvalidArgument("unknown file type: ", filename);
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

    OP_REQUIRES_OK(
        context, resource_->Init(input_tensor->scalar<string>()(), nullptr, 0));
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

class AudioDecodeWAVOp : public OpKernel {
 public:
  explicit AudioDecodeWAVOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->input("shape", &shape_tensor));

    const string& input = input_tensor->scalar<string>()();

    std::unique_ptr<AudioReadableResourceBase> resource;
    OP_REQUIRES_OK(context,
                   WAVReadableResourceInit(env_, "memory", input.data(),
                                           input.size(), resource));

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
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableInit").Device(DEVICE_CPU),
                        AudioReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableSpec").Device(DEVICE_CPU),
                        AudioReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableRead").Device(DEVICE_CPU),
                        AudioReadableReadOp);

REGISTER_KERNEL_BUILDER(Name("IO>AudioResample").Device(DEVICE_CPU),
                        AudioResampleOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioDecodeWAV").Device(DEVICE_CPU),
                        AudioDecodeWAVOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
