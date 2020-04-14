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
#include "tensorflow_io/core/kernels/io_stream.h"
#include <ImfMultiPartInputFile.h>
#include <ImfInputPart.h>
#include <ImfChannelList.h>

namespace tensorflow {
namespace data {
namespace {

class OpenEXRIStream : public OPENEXR_IMF_INTERNAL_NAMESPACE::IStream {
public:
  OpenEXRIStream(const string& filename, SizedRandomAccessFile* file, int64 size)
  : OPENEXR_IMF_INTERNAL_NAMESPACE::IStream(filename.c_str())
  , file_(file)
  , size_(size)
  , pos_(0) {}

  ~OpenEXRIStream() {}

  bool read (char c[/*n*/], int n) override {
    if (pos_ + n > size_) {
      throw IEX_NAMESPACE::InputExc("Unexpected end of file.");
    }
    StringPiece result;
    Status status = file_->Read(pos_, n, &result, c);
    if (!status.ok() || result.size() < n) {
      throw IEX_NAMESPACE::InputExc("Unexpected end of file.");
    }
    pos_ += n;
    return (pos_ < size_);
  }
  OPENEXR_IMF_INTERNAL_NAMESPACE::Int64 tellg () override {
    return pos_;
  }
  void seekg (OPENEXR_IMF_INTERNAL_NAMESPACE::Int64 pos) override {
    if (pos > size_) {
      throw IEX_NAMESPACE::InputExc("seekg beyond end of file");
    }
    pos_ = pos;
  }
 private:
  SizedRandomAccessFile *file_ = nullptr;
  OPENEXR_IMF_INTERNAL_NAMESPACE::Int64 size_ = 0;
  OPENEXR_IMF_INTERNAL_NAMESPACE::Int64 pos_ = 0;
};

class DecodeEXRInfoOp : public OpKernel {
 public:
  explicit DecodeEXRInfoOp(OpKernelConstruction* context)
   : OpKernel(context) {
     env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const string filename = "memory";
    string memory = input_tensor->scalar<tstring>()();
    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    OpenEXRIStream stream(filename, file.get(), size);
    Imf::MultiPartInputFile input_file(stream);

    std::vector<std::pair<int64, int64>> shape;
    std::vector<std::vector<int64>> dtype;
    std::vector<std::vector<string>> channel;
    int64 maxchannel = 0;
    for (int i = 0; i < input_file.parts(); i++) {
      Imath::Box2i dw = input_file.header(i).dataWindow();
      int64 height = dw.max.y - dw.min.y + 1;
      int64 width  = dw.max.x - dw.min.x + 1;
      shape.push_back(std::pair<int64, int64>(height, width));

      dtype.push_back(std::vector<int64>());
      channel.push_back(std::vector<string>());
      for (Imf::ChannelList::ConstIterator c = input_file.header(i).channels().begin(); c != input_file.header(i).channels().end(); c++) {
        switch (c.channel().type)
        {
        case Imf::UINT:
          dtype.back().push_back(DT_UINT32);
          break;
        case Imf::HALF:
          dtype.back().push_back(DT_HALF);
          break;
        case Imf::FLOAT:
          dtype.back().push_back(DT_FLOAT);
          break;
        default:
          OP_REQUIRES(context, false, errors::InvalidArgument("invalid data type: ", c.channel().type));
        }
        channel.back().push_back(c.name());
      }
      maxchannel = maxchannel > channel.back().size() ? maxchannel : channel.back().size();
    }

    Tensor* shape_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int64>(shape.size()), 2}), &shape_tensor));
    for (size_t i = 0; i < shape.size(); i++) {
      shape_tensor->flat<int64>()(i * 2) = shape[i].first;
      shape_tensor->flat<int64>()(i * 2 + 1) = shape[i].second;
    }

    Tensor* dtype_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({static_cast<int64>(dtype.size()), maxchannel}), &dtype_tensor));
    Tensor* channel_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({static_cast<int64>(dtype.size()), maxchannel}), &channel_tensor));
    for (size_t i = 0; i < dtype.size(); i++) {
      for (size_t j = 0; j < dtype[i].size(); j++) {
        dtype_tensor->flat<int64>()(i * maxchannel + j) = dtype[i][j];
        channel_tensor->flat<tstring>()(i * maxchannel + j) = channel[i][j];
      }
      for (size_t j = dtype[i].size(); j < maxchannel; j++) {
        dtype_tensor->flat<int64>()(i * maxchannel + j) = 0;
        channel_tensor->flat<tstring>()(i * maxchannel + j) = "";
      }
    }
  }
 private:
  mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};
class DecodeEXROp : public OpKernel {
 public:
  explicit DecodeEXROp(OpKernelConstruction* context)
   : OpKernel(context) {
     env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));

    const Tensor* channel_tensor;
    OP_REQUIRES_OK(context, context->input("channel", &channel_tensor));

    const string filename = "memory";
    string memory = input_tensor->scalar<tstring>()();
    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    OpenEXRIStream stream(filename, file.get(), size);
    Imf::MultiPartInputFile input_file(stream);

    const int64 index = index_tensor->scalar<int64>()();
    string channel = channel_tensor->scalar<tstring>()();

    Imf::InputPart input_part(input_file, index);

    Imath::Box2i dw = input_part.header().dataWindow();

    int64 height = dw.max.y - dw.min.y + 1;
    int64 width  = dw.max.x - dw.min.x + 1;

    DataType dtype = DT_INVALID;
    Imf::PixelType pixel_type = Imf::UINT;
    int x_sampling, y_sampling;
    for (Imf::ChannelList::ConstIterator c = input_part.header().channels().begin(); c != input_part.header().channels().end(); c++) {
      if (channel == c.name()) {
        pixel_type = c.channel().type;
        x_sampling = c.channel().xSampling;
        y_sampling = c.channel().ySampling;
        switch (pixel_type)
        {
        case Imf::UINT:
          dtype = DT_UINT32;
          break;
        case Imf::HALF:
          dtype = DT_HALF;
          break;
        case Imf::FLOAT:
          dtype = DT_FLOAT;
          break;
        default:
          OP_REQUIRES(context, false, errors::InvalidArgument("invalid pixel type: ", pixel_type));
        }
      }
    }
    OP_REQUIRES(context, (dtype != DT_INVALID), errors::InvalidArgument("unable to find channel: ", channel));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({height, width}), &output_tensor));

    char *base;
    size_t byte;
    switch (dtype)
    {
    case DT_UINT32:
      base = (char *)output_tensor->flat<uint32>().data();
      byte = sizeof(uint32);
      break;
    case DT_HALF:
      base = (char *)output_tensor->flat<Eigen::half>().data();
      byte = sizeof(Eigen::half);
      break;
    case DT_FLOAT:
      base = (char *)output_tensor->flat<float>().data();
      byte = sizeof(float);
      break;
    default:
      OP_REQUIRES(context, false, errors::InvalidArgument("invalid data type: ", dtype));
    }

    Imf::FrameBuffer frame_buffer;
    frame_buffer.insert(channel, Imf::Slice::Make(pixel_type, base, dw, 0, 0, x_sampling, y_sampling));

    input_part.setFrameBuffer(frame_buffer);
    input_part.readPixels(dw.min.y, dw.max.y);
  }
 private:
  mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodeExrInfo").Device(DEVICE_CPU),
                        DecodeEXRInfoOp);
REGISTER_KERNEL_BUILDER(Name("IO>DecodeExr").Device(DEVICE_CPU),
                        DecodeEXROp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
