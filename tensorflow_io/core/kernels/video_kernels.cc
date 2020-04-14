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

#include "tensorflow_io/core/kernels/video_kernels.h"

extern "C" {
#if defined(__APPLE__)
void* VideoCaptureInitFunction(const char* device, int64_t* bytes,
                               int64_t* width, int64_t* height);
void VideoCaptureNextFunction(void* context, void* data, int64_t size);
void VideoCaptureFiniFunction(void* context);
#elif defined(_MSC_VER)
void* VideoCaptureInitFunction(const char* device, int64_t* bytes,
                               int64_t* width, int64_t* height) {
  return NULL;
}
void VideoCaptureNextFunction(void* context, void* data, int64_t size) {}
void VideoCaptureFiniFunction(void* context) {}
#else
void* VideoCaptureInitFunction(const char* device, int64_t* bytes,
                               int64_t* width, int64_t* height) {
  tensorflow::data::VideoCaptureContext* p =
      new tensorflow::data::VideoCaptureContext();
  if (p != nullptr) {
    tensorflow::Status status = p->Init(device, bytes, width, height);
    if (status.ok()) {
      return p;
    }
    LOG(ERROR) << "unable to initialize video capture: " << status;
    delete p;
  }
  return NULL;
}
void VideoCaptureNextFunction(void* context, void* data, int64_t size) {
  tensorflow::data::VideoCaptureContext* p =
      static_cast<tensorflow::data::VideoCaptureContext*>(context);
  if (p != nullptr) {
    tensorflow::Status status = p->Read(data, size);
    if (!status.ok()) {
      LOG(ERROR) << "unable to read video capture: " << status;
    }
  }
}
void VideoCaptureFiniFunction(void* context) {
  tensorflow::data::VideoCaptureContext* p =
      static_cast<tensorflow::data::VideoCaptureContext*>(context);
  if (p != nullptr) {
    delete p;
  }
}
#endif
}
namespace tensorflow {
namespace data {
namespace {

class VideoCaptureReadableResource : public ResourceBase {
 public:
  VideoCaptureReadableResource(Env* env)
      : env_(env), context_(nullptr, [](void* p) {
          if (p != nullptr) {
            VideoCaptureFiniFunction(p);
          }
        }) {}
  ~VideoCaptureReadableResource() {}

  Status Init(const string& input) {
    mutex_lock l(mu_);

    int64_t bytes, width, height;
    context_.reset(
        VideoCaptureInitFunction(input.c_str(), &bytes, &width, &height));
    if (context_.get() == nullptr) {
      return errors::InvalidArgument("unable to open device ", input);
    }
    bytes_ = static_cast<int64>(bytes);
    width_ = static_cast<int64>(width);
    height_ = static_cast<int64>(height);
    return Status::OK();
  }
  Status Read(
      std::function<Status(const TensorShape& shape, Tensor** value_tensor)>
          allocate_func) {
    mutex_lock l(mu_);

    Tensor* value_tensor;
    TF_RETURN_IF_ERROR(allocate_func(TensorShape({1}), &value_tensor));

    string buffer;
    buffer.resize(bytes_);
    VideoCaptureNextFunction(context_.get(), (void*)&buffer[0],
                             static_cast<int64_t>(bytes_));
    value_tensor->flat<tstring>()(0) = buffer;

    return Status::OK();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return "VideoCaptureReadableResource";
  }

 protected:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);

  std::unique_ptr<void, void (*)(void*)> context_;
  int64 bytes_;
  int64 width_;
  int64 height_;
};

class VideoCaptureReadableInitOp
    : public ResourceOpKernel<VideoCaptureReadableResource> {
 public:
  explicit VideoCaptureReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<VideoCaptureReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<VideoCaptureReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(context, resource_->Init(input));
  }
  Status CreateResource(VideoCaptureReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new VideoCaptureReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class VideoCaptureReadableReadOp : public OpKernel {
 public:
  explicit VideoCaptureReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    VideoCaptureReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    OP_REQUIRES_OK(
        context, resource->Read([&](const TensorShape& shape,
                                    Tensor** value_tensor) -> Status {
          TF_RETURN_IF_ERROR(context->allocate_output(0, shape, value_tensor));
          return Status::OK();
        }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>VideoCaptureReadableInit").Device(DEVICE_CPU),
                        VideoCaptureReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>VideoCaptureReadableRead").Device(DEVICE_CPU),
                        VideoCaptureReadableReadOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
