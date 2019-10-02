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

#include "tensorflow/core/framework/op_kernel.h"
#include <tiffio.h>
#include <tiffio.hxx>

namespace tensorflow {
namespace data {
namespace {
class DecodeTIFFInfoOp : public OpKernel {
 public:
  explicit DecodeTIFFInfoOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    std::istringstream input_stream(input_tensor->scalar<string>()(), std::ios_base::in | std::ios_base::binary);

    std::unique_ptr<TIFF, void(*)(TIFF*)> tiff(TIFFStreamOpen("memory", &input_stream), [](TIFF* p) { if (p != nullptr) { TIFFClose(p); } });
    OP_REQUIRES(context, (tiff.get() != nullptr), errors::InvalidArgument("unable to open TIFF from memory"));

    std::vector<std::pair<int64, int64>> shape;
    do {
      unsigned int height, width;
      TIFFGetField(tiff.get(), TIFFTAG_IMAGELENGTH, &height);
      TIFFGetField(tiff.get(), TIFFTAG_IMAGEWIDTH, &width);
      shape.push_back(std::pair<int64, int64>(static_cast<int64>(height), static_cast<int64>(width)));
    } while (TIFFReadDirectory(tiff.get()));

    Tensor* shape_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int64>(shape.size()), 2}), &shape_tensor));
    for (size_t i = 0; i < shape.size(); i++) {
      shape_tensor->flat<int64>()(i * 2) = shape[i].first;
      shape_tensor->flat<int64>()(i * 2 + 1) = shape[i].second;
    }
  }
};
class DecodeTIFFOp : public OpKernel {
 public:
  explicit DecodeTIFFOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));

    std::istringstream input_stream(input_tensor->scalar<string>()(), std::ios_base::in | std::ios_base::binary);

    std::unique_ptr<TIFF, void(*)(TIFF*)> tiff(TIFFStreamOpen("memory", &input_stream), [](TIFF* p) { if (p != nullptr) { TIFFClose(p); } });
    OP_REQUIRES(context, (tiff.get() != nullptr), errors::InvalidArgument("unable to open TIFF from memory"));


    int status = TIFFSetDirectory(tiff.get(), index_tensor->scalar<int64>()());
    OP_REQUIRES(context, (status), errors::InvalidArgument("unable to set TIFF directory to ", index_tensor->scalar<int64>()()));
    unsigned int height, width;
    TIFFGetField(tiff.get(), TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tiff.get(), TIFFTAG_IMAGEWIDTH, &width);

    Tensor* image_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int64>(height), static_cast<int64>(width), channels_}), &image_tensor));

    uint32* raster = reinterpret_cast<uint32*>(image_tensor->flat<uint8>().data());
    OP_REQUIRES(context, (TIFFReadRGBAImageOriented(tiff.get(), width, height, raster, ORIENTATION_TOPLEFT, 0)),errors::InvalidArgument("unable to read directory: ", index_tensor->scalar<int64>()()));
  }

 private:
  // TODO (yongtang): Set channels_ = 4 for now.
  static const int channels_ = 4;
};
REGISTER_KERNEL_BUILDER(Name("DecodeTiffInfo").Device(DEVICE_CPU),
                        DecodeTIFFInfoOp);
REGISTER_KERNEL_BUILDER(Name("DecodeTiff").Device(DEVICE_CPU),
                        DecodeTIFFOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
