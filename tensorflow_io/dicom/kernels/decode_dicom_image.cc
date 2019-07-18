/* Copyright 2019 Gradient Health Inc. All Rights Reserved.
   Author: Marcelo Lerendegui <marcelo@gradienthealth.io>

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

#include "dcmtk/config/osconfig.h"

#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/ofstd/ofstring.h"
#include "dcmtk/ofstd/ofstdinc.h"
#include "dcmtk/ofstd/oftypes.h"
#include <dcmtk/dcmdata/dcfilefo.h>
#include "dcmtk/dcmdata/dcistrmb.h"
#include "dcmtk/dcmdata/dcdict.h"

#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmdata/dcrledrg.h"  /* for DcmRLEDecoderRegistration */
#include "dcmtk/dcmjpeg/djdecode.h"  /* for dcmjpeg decoders */
#include "dcmtk/dcmjpeg/dipijpeg.h"  /* for dcmimage JPEG plugin */
#include "dcmtk/dcmjpls/djdecode.h"  /* for dcmjpls decoders */
#include "dcmtk/dcmimage/dipitiff.h" /* for dcmimage TIFF plugin */
#include "dcmtk/dcmimage/dipipng.h"  /* for dcmimage PNG plugin */
#include "dcmtk/dcmimage/diregist.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include <cstdint>
#include <exception>

typedef uint64_t Uint64; // Uint64 not present in tensorflow::custom-op docker image dcmtk

using namespace tensorflow;

template <typename dtype>
class DecodeDICOMImageOp : public OpKernel
{
public:
    explicit DecodeDICOMImageOp(OpKernelConstruction *context) : OpKernel(context)
    {
        // Get the on_error
        OP_REQUIRES_OK(context, context->GetAttr("on_error", &on_error));

        // Get the on_error
        OP_REQUIRES_OK(context, context->GetAttr("scale", &scale));

        // Get the color_dim
        OP_REQUIRES_OK(context, context->GetAttr("color_dim", &color_dim));

        DcmRLEDecoderRegistration::registerCodecs(); // register RLE codecs
        DJDecoderRegistration::registerCodecs();     // register JPEG codecs
        DJLSDecoderRegistration::registerCodecs();   // register JPEG-LS codecs
    }

    ~DecodeDICOMImageOp()
    {
        DcmRLEDecoderRegistration::cleanup(); // deregister RLE codecs
        DJDecoderRegistration::cleanup();     // deregister JPEG codecs
        DJLSDecoderRegistration::cleanup();   // deregister JPEG-LS codecs
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input file content tensor
        const Tensor &in_contents = context->input(0);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(in_contents.shape()),
                    errors::InvalidArgument(
                        "DecodeDICOMImage expects input content tensor to be scalar, but had shape: ",
                        in_contents.shape().DebugString()));

        const auto in_contents_scalar = in_contents.scalar<string>()();

        // Load Dicom Image
        DcmInputBufferStream dataBuf;
        dataBuf.setBuffer(in_contents_scalar.data(), in_contents_scalar.length());
        dataBuf.setEos();

        DcmFileFormat *dfile = new DcmFileFormat();
        dfile->transferInit();
        OFCondition cond = dfile->read(dataBuf);
        dfile->transferEnd();

        DicomImage *image = NULL;
        try
        {
            image = new DicomImage(dfile, EXS_Unknown, CIF_DecompressCompletePixelData);
        }
        catch (...)
        {
            image = NULL;
        }

        unsigned long frameWidth = 0;
        unsigned long frameHeight = 0;
        unsigned int pixelDepth = 0;
        unsigned long dataSize = 0;
        unsigned long frameCount = 0;
        unsigned int samples_per_pixel = 0;

        if ((image == NULL) || (image->getStatus() != EIS_Normal))
        {
            if (on_error == "strict")
            {
                OP_REQUIRES(context, false,
                            errors::InvalidArgument("Error loading image"));
                return;
            }
            else if ((on_error == "skip") || (on_error == "lossy"))
            {
                Tensor *output_tensor = NULL;
                OP_REQUIRES_OK(context, context->allocate_output(0,
                                                                 {0},
                                                                 &output_tensor));
                return;
            }
        }

        // Get image information
        frameCount = image->getFrameCount(); // getNumberOfFrames(); starts at version DCMTK-3.6.1_20140617
        frameWidth = image->getWidth();
        frameHeight = image->getHeight();
        pixelDepth = image->getDepth();
        samples_per_pixel = image->isMonochrome() ? 1 : 3;

        // Create an output tensor shape
        TensorShape out_shape;
        if ((samples_per_pixel == 1) && (color_dim == false))
        {
            out_shape = TensorShape({frameCount, frameHeight, frameWidth});
        }
        else
        {
            out_shape = TensorShape({frameCount, frameHeight, frameWidth, samples_per_pixel});
        }

        // Check if output type is ok for image
        if (pixelDepth > sizeof(dtype) * 8)
        {
            if (on_error == "strict")
            {
                OP_REQUIRES(context, false,
                            errors::InvalidArgument("Input argument dtype size smaller than pixelDepth (bits):", pixelDepth));
                return;
            }
            else if (on_error == "skip")
            {
                Tensor *output_tensor = NULL;
                OP_REQUIRES_OK(context, context->allocate_output(0,
                                                                 {0},
                                                                 &output_tensor));
                return;
            }
        }

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                                                         out_shape,
                                                         &output_tensor));

        auto output_flat = output_tensor->template flat<dtype>();

        unsigned long frame_pixel_count = frameHeight * frameWidth * samples_per_pixel;

        for (Uint64 f = 0; f < frameCount; f++)
        {
            const void *image_frame = image->getOutputData(pixelDepth, f);

            for (Uint64 p = 0; p < frame_pixel_count; p++)
            {
                output_flat(f * frame_pixel_count + p) = convert_uintn_to_t(image_frame, pixelDepth, p);
            }
        }
        delete image;
    }

    dtype convert_uintn_to_t(const void *buff, unsigned int n_bits, unsigned int pos)
    {
        uint64 in_value;
        if (n_bits <= 8)
            in_value = *((const uint8 *)buff + pos);
        else if (n_bits <= 16)
            in_value = *((const uint16 *)buff + pos);
        else if (n_bits <= 32)
            in_value = *((const uint32 *)buff + pos);
        else
            in_value = *((const uint64 *)buff + pos);

        dtype out;
        uint64_to_t(in_value, n_bits, &out);
        return out;
    }

    void uint64_to_t(uint64 in_value, unsigned int n_bits, uint8 *out_value)
    {
        if (scale == "auto")
        {
            in_value = in_value << (64 - n_bits);
            *out_value = (dtype)(in_value >> (64 - 8 * sizeof(uint8)));
        }
        else if (scale == "preserve")
        {
            *out_value = in_value >= ((1ULL << 8 * sizeof(uint8)) - 1) ? (dtype)(((1ULL << 8 * sizeof(uint8)) - 1)) : (dtype)(in_value);
        }
    }

    void uint64_to_t(uint64 in_value, unsigned int n_bits, uint16 *out_value)
    {
        if (scale == "auto")
        {
            in_value = in_value << (64 - n_bits);
            *out_value = (dtype)(in_value >> (64 - 8 * sizeof(uint16)));
        }
        else if (scale == "preserve")
        {
            *out_value = in_value >= ((1ULL << 8 * sizeof(uint16)) - 1) ? (dtype)(((1ULL << 8 * sizeof(uint16)) - 1)) : (dtype)(in_value);
        }
    }

    void uint64_to_t(uint64 in_value, unsigned int n_bits, uint32 *out_value)
    {
        if (scale == "auto")
        {
            in_value = in_value << (64 - n_bits);
            *out_value = (dtype)(in_value >> (64 - 8 * sizeof(uint32)));
        }
        else if (scale == "preserve")
        {
            *out_value = in_value >= ((1ULL << 8 * sizeof(uint32)) - 1) ? (dtype)(((1ULL << 8 * sizeof(uint32)) - 1)) : (dtype)(in_value);
        }
    }

    void uint64_to_t(uint64 in_value, unsigned int n_bits, uint64 *out_value)
    {
        *out_value = in_value;
    }

    void uint64_to_t(uint64 in_value, unsigned int n_bits, float *out_value)
    {
        if (scale == "auto")
            *out_value = (float)(in_value) / (float)((1ULL << n_bits) - 1);
        else if (scale == "preserve")
            *out_value = (float)(in_value);
    }

    void uint64_to_t(uint64 in_value, unsigned int n_bits, Eigen::half *out_value)
    {
        if (scale == "auto")
            *out_value = static_cast<Eigen::half>((double)(in_value) / (double)((1ULL << n_bits) - 1));
        else if (scale == "preserve")
            *out_value = static_cast<Eigen::half>(in_value);
    }

    void uint64_to_t(uint64 in_value, unsigned int n_bits, double *out_value)
    {
        if (scale == "auto")
            *out_value = (double)(in_value) / (double)((1ULL << n_bits) - 1);
        else if (scale == "preserve")
            *out_value = (double)(in_value);
    }

    string on_error;
    string scale;
    bool color_dim;
};

// Register the CPU kernels.
#define REGISTER_DECODE_DICOM_IMAGE_CPU(dtype)                                      \
    REGISTER_KERNEL_BUILDER(                                                        \
        Name("DecodeDICOMImage").Device(DEVICE_CPU).TypeConstraint<dtype>("dtype"), \
        DecodeDICOMImageOp<dtype>);

REGISTER_DECODE_DICOM_IMAGE_CPU(uint8);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint16);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint32);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint64);
REGISTER_DECODE_DICOM_IMAGE_CPU(float);
REGISTER_DECODE_DICOM_IMAGE_CPU(Eigen::half);
REGISTER_DECODE_DICOM_IMAGE_CPU(double);

#undef REGISTER_DECODE_DICOM_IMAGE_CPU
