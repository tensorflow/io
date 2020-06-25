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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {
namespace data {
namespace {

class FileInfoOp : public OpKernel {
 public:
  explicit FileInfoOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    string input = input_tensor->scalar<tstring>()();

    uint64 size;
    OP_REQUIRES_OK(context, env_->GetFileSize(input, &size));

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    OP_REQUIRES_OK(context, env_->NewRandomAccessFile(input, &file));

    Tensor* length_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &length_tensor));
    length_tensor->scalar<int64>()() = size;

    Tensor* compression_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &compression_tensor));

    StringPiece result;
    char buffer[10] = {0};
    Status status = file->Read(0, 10, &result, buffer);
    if (!status.ok() || result.size() != 10) {
      return;
    }
    // deflation- third byte must be 0x08.
    if (memcmp(buffer, "\x1F\x8B\x08", 3) != 0) {
      return;
    }
    // No reserved flags set.
    if ((buffer[3] & 0xE0) != 0) {
      return;
    }
    compression_tensor->scalar<tstring>()() = "GZIP";
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class FileReadOp : public OpKernel {
 public:
  explicit FileReadOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<tstring>()();

    const Tensor* offset_tensor;
    OP_REQUIRES_OK(context, context->input("offset", &offset_tensor));
    const int64 offset = offset_tensor->scalar<int64>()();

    const Tensor* length_tensor;
    OP_REQUIRES_OK(context, context->input("length", &length_tensor));
    const int64 length = length_tensor->scalar<int64>()();

    const Tensor* compression_tensor;
    OP_REQUIRES_OK(context, context->input("compression", &compression_tensor));
    const string& compression = compression_tensor->scalar<tstring>()();

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    OP_REQUIRES_OK(context, env_->NewRandomAccessFile(input, &file));

    std::unique_ptr<tensorflow::io::RandomAccessInputStream> input_stream(
        new tensorflow::io::RandomAccessInputStream(file.get()));

    tensorflow::io::InputStreamInterface* stream = input_stream.get();

    std::unique_ptr<tensorflow::io::ZlibInputStream> zlib_stream;
    if (compression == "GZIP") {
      zlib_stream.reset(new tensorflow::io::ZlibInputStream(
          input_stream.get(), 256 * 1024, 256 * 1024,
          tensorflow::io::ZlibCompressionOptions::GZIP()));
      stream = zlib_stream.get();
    }

    OP_REQUIRES_OK(context, stream->SkipNBytes(offset));

    tstring value;
    OP_REQUIRES_OK(context, stream->ReadNBytes(length, &value));

    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &value_tensor));
    value_tensor->scalar<tstring>()() = value;
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class FileResource : public ResourceBase {
 public:
  FileResource(Env* env) : env_(env) {}
  ~FileResource() {
    if (file_.get() != nullptr) {
      file_->Close();
    }
  }

  Status Init(const string& filename) {
    TF_RETURN_IF_ERROR(env_->NewWritableFile(filename, &file_));
    return Status::OK();
  }
  Status Write(const Tensor& content) {
    mutex_lock l(mu_);
    for (int64 i = 0; i < content.NumElements(); i++) {
      TF_RETURN_IF_ERROR(file_->Append(content.flat<tstring>()(i)));
    }
    return Status::OK();
  }
  Status Sync() {
    file_->Flush();
    return Status::OK();
  }
  Status Close() {
    file_.reset(nullptr);
    return Status::OK();
  }
  string DebugString() const override { return "FileResource"; }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<WritableFile> file_;
};

class FileInitOp : public ResourceOpKernel<FileResource> {
 public:
  explicit FileInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<FileResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<FileResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    OP_REQUIRES_OK(context, resource_->Init(input_tensor->scalar<tstring>()()));
  }
  Status CreateResource(FileResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new FileResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class FileCallOp : public OpKernel {
 public:
  explicit FileCallOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* final_tensor;
    OP_REQUIRES_OK(context, context->input("final", &final_tensor));

    FileResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    OP_REQUIRES_OK(context, resource->Write(*input_tensor));

    if (final_tensor->scalar<bool>()()) {
      OP_REQUIRES_OK(context, resource->Close());
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class FileSyncOp : public OpKernel {
 public:
  explicit FileSyncOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    FileResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);
    OP_REQUIRES_OK(context, resource->Sync());
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>FileInfo").Device(DEVICE_CPU), FileInfoOp);
REGISTER_KERNEL_BUILDER(Name("IO>FileRead").Device(DEVICE_CPU), FileReadOp);

REGISTER_KERNEL_BUILDER(Name("IO>FileInit").Device(DEVICE_CPU), FileInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>FileCall").Device(DEVICE_CPU), FileCallOp);
REGISTER_KERNEL_BUILDER(Name("IO>FileSync").Device(DEVICE_CPU), FileSyncOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
