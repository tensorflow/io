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
#include <dcmtk/dcmdata/dcfilefo.h>
#include "dcmtk/dcmdata/dcistrmb.h"
#include "dcmtk/dcmdata/dcdict.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class DecodeDICOMDataOp : public OpKernel
{
public:
    explicit DecodeDICOMDataOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    ~DecodeDICOMDataOp()
    {
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input file content tensor
        const Tensor &in_contents = context->input(0);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(in_contents.shape()),
                    errors::InvalidArgument(
                        "DecodeDICOMData expects input content tensor to be scalar, but had shape: ",
                        in_contents.shape().DebugString()));

        const auto in_contents_scalar = in_contents.scalar<string>()();

        const Tensor *in_tags;
        OP_REQUIRES_OK(context, context->input("tags", &in_tags));
        auto in_tags_flat = in_tags->flat<uint32>();

        // Create an output tensor
        Tensor *out_tag_values = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                                                         in_tags->shape(),
                                                         &out_tag_values));

        auto out_tag_values_flat = out_tag_values->flat<string>();

        DcmInputBufferStream dataBuf;
        dataBuf.setBuffer(in_contents_scalar.data(), in_contents_scalar.length());
        dataBuf.setEos();

        DcmFileFormat dfile;
        dfile.transferInit();
        OFCondition cond = dfile.read(dataBuf);
        dfile.transferEnd();

        DcmDataset *dset = dfile.getDataset();
        DcmMetaInfo *meta = dfile.getMetaInfo();

        for (unsigned int tag_i = 0; tag_i < in_tags_flat.size(); tag_i++)
        {
            uint32 tag_value = in_tags_flat(tag_i);
            uint16 tag_group_number = (uint16)((tag_value & 0xFFFF0000) >> 16);
            uint16 tag_element_number = (uint16)((tag_value & 0x0000FFFF) >> 0);
            DcmTag tag(tag_group_number, tag_element_number);

            OFString val;
            if (dset->tagExists(tag))
                dset->findAndGetOFStringArray(tag, val);
            else if (meta->tagExists(tag))
                meta->findAndGetOFStringArray(tag, val);
            else
                val = OFString("");

            out_tag_values_flat(tag_i) = val.c_str();
        }
    }
};
REGISTER_KERNEL_BUILDER(Name("DecodeDICOMData").Device(DEVICE_CPU), DecodeDICOMDataOp);