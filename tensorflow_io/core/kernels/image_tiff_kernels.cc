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

#include <tiffio.h>
#include <xtiffio.h>

#include <tiffio.hxx>

#include "geotiff.h"
#include "tensorflow/core/framework/op_kernel.h"

// Repackge XTIFFStreamOpen from TIFFStreamOpen in libtiff/tif_stream.cxx
extern "C" {
using namespace std;

struct tiffis_data;
struct tiffos_data;

static tmsize_t _tiffosReadProc(thandle_t, void*, tmsize_t);
static tmsize_t _tiffisReadProc(thandle_t fd, void* buf, tmsize_t size);
static tmsize_t _tiffosWriteProc(thandle_t fd, void* buf, tmsize_t size);
static tmsize_t _tiffisWriteProc(thandle_t, void*, tmsize_t);
static uint64 _tiffosSeekProc(thandle_t fd, uint64 off, int whence);
static uint64 _tiffisSeekProc(thandle_t fd, uint64 off, int whence);
static uint64 _tiffosSizeProc(thandle_t fd);
static uint64 _tiffisSizeProc(thandle_t fd);
static int _tiffosCloseProc(thandle_t fd);
static int _tiffisCloseProc(thandle_t fd);
static int _tiffDummyMapProc(thandle_t, void** base, toff_t* size);
static void _tiffDummyUnmapProc(thandle_t, void* base, toff_t size);

struct tiffis_data {
  istream* stream;
  ios::pos_type start_pos;
};

struct tiffos_data {
  ostream* stream;
  ios::pos_type start_pos;
};

static tmsize_t _tiffosReadProc(thandle_t, void*, tmsize_t) { return 0; }

static tmsize_t _tiffisReadProc(thandle_t fd, void* buf, tmsize_t size) {
  tiffis_data* data = reinterpret_cast<tiffis_data*>(fd);

  // Verify that type does not overflow.
  streamsize request_size = size;
  if (static_cast<tmsize_t>(request_size) != size)
    return static_cast<tmsize_t>(-1);

  data->stream->read((char*)buf, request_size);

  return static_cast<tmsize_t>(data->stream->gcount());
}

static tmsize_t _tiffosWriteProc(thandle_t fd, void* buf, tmsize_t size) {
  tiffos_data* data = reinterpret_cast<tiffos_data*>(fd);
  ostream* os = data->stream;
  ios::pos_type pos = os->tellp();

  // Verify that type does not overflow.
  streamsize request_size = size;
  if (static_cast<tmsize_t>(request_size) != size)
    return static_cast<tmsize_t>(-1);

  os->write(reinterpret_cast<const char*>(buf), request_size);

  return static_cast<tmsize_t>(os->tellp() - pos);
}

static tmsize_t _tiffisWriteProc(thandle_t, void*, tmsize_t) { return 0; }

static uint64 _tiffosSeekProc(thandle_t fd, uint64 off, int whence) {
  tiffos_data* data = reinterpret_cast<tiffos_data*>(fd);
  ostream* os = data->stream;

  // if the stream has already failed, don't do anything
  if (os->fail()) return static_cast<uint64>(-1);

  switch (whence) {
    case SEEK_SET: {
      // Compute 64-bit offset
      uint64 new_offset = static_cast<uint64>(data->start_pos) + off;

      // Verify that value does not overflow
      ios::off_type offset = static_cast<ios::off_type>(new_offset);
      if (static_cast<uint64>(offset) != new_offset)
        return static_cast<uint64>(-1);

      os->seekp(offset, ios::beg);
      break;
    }
    case SEEK_CUR: {
      // Verify that value does not overflow
      ios::off_type offset = static_cast<ios::off_type>(off);
      if (static_cast<uint64>(offset) != off) return static_cast<uint64>(-1);

      os->seekp(offset, ios::cur);
      break;
    }
    case SEEK_END: {
      // Verify that value does not overflow
      ios::off_type offset = static_cast<ios::off_type>(off);
      if (static_cast<uint64>(offset) != off) return static_cast<uint64>(-1);

      os->seekp(offset, ios::end);
      break;
    }
  }

  // Attempt to workaround problems with seeking past the end of the
  // stream.  ofstream doesn't have a problem with this but
  // ostrstream/ostringstream does. In that situation, add intermediate
  // '\0' characters.
  if (os->fail()) {
#ifdef __VMS
    int old_state;
#else
    ios::iostate old_state;
#endif
    ios::pos_type origin;

    old_state = os->rdstate();
    // reset the fail bit or else tellp() won't work below
    os->clear(os->rdstate() & ~ios::failbit);
    switch (whence) {
      case SEEK_SET:
      default:
        origin = data->start_pos;
        break;
      case SEEK_CUR:
        origin = os->tellp();
        break;
      case SEEK_END:
        os->seekp(0, ios::end);
        origin = os->tellp();
        break;
    }
    // restore original stream state
    os->clear(old_state);

    // only do something if desired seek position is valid
    if ((static_cast<uint64>(origin) + off) >
        static_cast<uint64>(data->start_pos)) {
      uint64 num_fill;

      // clear the fail bit
      os->clear(os->rdstate() & ~ios::failbit);

      // extend the stream to the expected size
      os->seekp(0, ios::end);
      num_fill = (static_cast<uint64>(origin)) + off - os->tellp();
      for (uint64 i = 0; i < num_fill; i++) os->put('\0');

      // retry the seek
      os->seekp(static_cast<ios::off_type>(static_cast<uint64>(origin) + off),
                ios::beg);
    }
  }

  return static_cast<uint64>(os->tellp());
}

static uint64 _tiffisSeekProc(thandle_t fd, uint64 off, int whence) {
  tiffis_data* data = reinterpret_cast<tiffis_data*>(fd);

  switch (whence) {
    case SEEK_SET: {
      // Compute 64-bit offset
      uint64 new_offset = static_cast<uint64>(data->start_pos) + off;

      // Verify that value does not overflow
      ios::off_type offset = static_cast<ios::off_type>(new_offset);
      if (static_cast<uint64>(offset) != new_offset)
        return static_cast<uint64>(-1);

      data->stream->seekg(offset, ios::beg);
      break;
    }
    case SEEK_CUR: {
      // Verify that value does not overflow
      ios::off_type offset = static_cast<ios::off_type>(off);
      if (static_cast<uint64>(offset) != off) return static_cast<uint64>(-1);

      data->stream->seekg(offset, ios::cur);
      break;
    }
    case SEEK_END: {
      // Verify that value does not overflow
      ios::off_type offset = static_cast<ios::off_type>(off);
      if (static_cast<uint64>(offset) != off) return static_cast<uint64>(-1);

      data->stream->seekg(offset, ios::end);
      break;
    }
  }

  return (uint64)(data->stream->tellg() - data->start_pos);
}

static uint64 _tiffosSizeProc(thandle_t fd) {
  tiffos_data* data = reinterpret_cast<tiffos_data*>(fd);
  ostream* os = data->stream;
  ios::pos_type pos = os->tellp();
  ios::pos_type len;

  os->seekp(0, ios::end);
  len = os->tellp();
  os->seekp(pos);

  return (uint64)len;
}

static uint64 _tiffisSizeProc(thandle_t fd) {
  tiffis_data* data = reinterpret_cast<tiffis_data*>(fd);
  ios::pos_type pos = data->stream->tellg();
  ios::pos_type len;

  data->stream->seekg(0, ios::end);
  len = data->stream->tellg();
  data->stream->seekg(pos);

  return (uint64)len;
}

static int _tiffosCloseProc(thandle_t fd) {
  // Our stream was not allocated by us, so it shouldn't be closed by us.
  delete reinterpret_cast<tiffos_data*>(fd);
  return 0;
}

static int _tiffisCloseProc(thandle_t fd) {
  // Our stream was not allocated by us, so it shouldn't be closed by us.
  delete reinterpret_cast<tiffis_data*>(fd);
  return 0;
}

static int _tiffDummyMapProc(thandle_t, void** base, toff_t* size) {
  (void)base;
  (void)size;
  return (0);
}

static void _tiffDummyUnmapProc(thandle_t, void* base, toff_t size) {
  (void)base;
  (void)size;
}

TIFF* XTIFFStreamOpen(const char* name, std::istream* is) {
  const char* mode = "rm";
  TIFF* tif;

  tiffis_data* data = new tiffis_data;
  data->stream = is;
  data->start_pos = data->stream->tellg();
  // Open for reading.
  tif = XTIFFClientOpen(name, mode, reinterpret_cast<thandle_t>(data),
                        _tiffisReadProc, _tiffisWriteProc, _tiffisSeekProc,
                        _tiffisCloseProc, _tiffisSizeProc, _tiffDummyMapProc,
                        _tiffDummyUnmapProc);
  if (!tif) {
    delete data;
  }
  return (tif);
}
}
namespace tensorflow {
namespace data {
namespace {

class DecodeTIFFInfoOp : public OpKernel {
 public:
  explicit DecodeTIFFInfoOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    std::istringstream input_stream(input_tensor->scalar<tstring>()(),
                                    std::ios_base::in | std::ios_base::binary);

    std::unique_ptr<TIFF, void (*)(TIFF*)> tiff(
        XTIFFStreamOpen("memory", &input_stream), [](TIFF* p) {
          if (p != nullptr) {
            XTIFFClose(p);
          }
        });
    OP_REQUIRES(context, (tiff.get() != nullptr),
                errors::InvalidArgument("unable to open TIFF from memory"));

    std::vector<TensorShape> shape;
    std::vector<DataType> dtype;
    do {
      unsigned int height, width;
      TIFFGetField(tiff.get(), TIFFTAG_IMAGELENGTH, &height);
      TIFFGetField(tiff.get(), TIFFTAG_IMAGEWIDTH, &width);

      unsigned short channels;
      TIFFGetField(tiff.get(), TIFFTAG_SAMPLESPERPIXEL, &channels);

      shape.push_back(
          TensorShape({static_cast<int64>(height), static_cast<int64>(width),
                       static_cast<int64>(channels)}));

      unsigned short format, bits;
      if (!TIFFGetField(tiff.get(), TIFFTAG_SAMPLEFORMAT, &format)) {
        // If format is not defined, then we assume format is SAMPLEFORMAT_UINT
        format = SAMPLEFORMAT_UINT;
      }
      TIFFGetField(tiff.get(), TIFFTAG_BITSPERSAMPLE, &bits);
      DataType pixel_dtype;
      switch (format) {
        case SAMPLEFORMAT_UINT:
          switch (bits) {
            case 8:
              pixel_dtype = DT_UINT8;
              break;
            case 16:
              pixel_dtype = DT_UINT16;
              break;
            default:
              OP_REQUIRES(context, false,
                          errors::InvalidArgument("unsupported bits ", bits,
                                                  " for uint"));
          }
          break;
        case SAMPLEFORMAT_INT:
          switch (bits) {
            default:
              OP_REQUIRES(context, false,
                          errors::InvalidArgument("unsupported bits ", bits,
                                                  " for int"));
          }
          break;
        case SAMPLEFORMAT_IEEEFP:
          switch (bits) {
            case 16:
              pixel_dtype = DT_HALF;
              break;
            default:
              OP_REQUIRES(context, false,
                          errors::InvalidArgument("unsupported bits ", bits,
                                                  " for fp"));
          }
          break;
        default:
          OP_REQUIRES(context, false,
                      errors::InvalidArgument("unsupported format ", format));
      }
      dtype.push_back(pixel_dtype);

      // GeoTIFF specifi information
      // std::unique_ptr<GTIF, void(*)(GTIF*)> gtif(GTIFNew(tiff.get()),
      // [](GTIF* p) { if (p != nullptr) { GTIFFree(p); } });
      // OP_REQUIRES(context, (gtif.get() != nullptr),
      // errors::InvalidArgument("unable to read GeoTIFF information"));
      // GTIFPrint(gtif,0,0);
    } while (TIFFReadDirectory(tiff.get()));

    Tensor* shape_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({static_cast<int64>(shape.size()), 3}),
                       &shape_tensor));
    for (size_t i = 0; i < shape.size(); i++) {
      shape_tensor->flat<int64>()(i * 3) = shape[i].dim_size(0);
      shape_tensor->flat<int64>()(i * 3 + 1) = shape[i].dim_size(1);
      shape_tensor->flat<int64>()(i * 3 + 2) = shape[i].dim_size(2);
    }
    Tensor* dtype_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            1, TensorShape({static_cast<int64>(dtype.size())}), &dtype_tensor));
    for (size_t i = 0; i < dtype.size(); i++) {
      dtype_tensor->flat<int64>()(i) = dtype[i];
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

    std::istringstream input_stream(input_tensor->scalar<tstring>()(),
                                    std::ios_base::in | std::ios_base::binary);

    std::unique_ptr<TIFF, void (*)(TIFF*)> tiff(
        XTIFFStreamOpen("memory", &input_stream), [](TIFF* p) {
          if (p != nullptr) {
            XTIFFClose(p);
          }
        });
    OP_REQUIRES(context, (tiff.get() != nullptr),
                errors::InvalidArgument("unable to open TIFF from memory"));

    int status = TIFFSetDirectory(tiff.get(), index_tensor->scalar<int64>()());
    OP_REQUIRES(context, (status),
                errors::InvalidArgument("unable to set TIFF directory to ",
                                        index_tensor->scalar<int64>()()));
    unsigned int height, width;
    TIFFGetField(tiff.get(), TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tiff.get(), TIFFTAG_IMAGEWIDTH, &width);

    Tensor* image_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0,
                       TensorShape({static_cast<int64>(height),
                                    static_cast<int64>(width), channels_}),
                       &image_tensor));

    uint32* raster =
        reinterpret_cast<uint32*>(image_tensor->flat<uint8>().data());
    OP_REQUIRES(context,
                (TIFFReadRGBAImageOriented(tiff.get(), width, height, raster,
                                           ORIENTATION_TOPLEFT, 0)),
                errors::InvalidArgument("unable to read directory: ",
                                        index_tensor->scalar<int64>()()));
  }

 private:
  // TODO (yongtang): Set channels_ = 4 for now.
  static const int channels_ = 4;
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodeTiffInfo").Device(DEVICE_CPU),
                        DecodeTIFFInfoOp);
REGISTER_KERNEL_BUILDER(Name("IO>DecodeTiff").Device(DEVICE_CPU), DecodeTIFFOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
