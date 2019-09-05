#include <vector>
#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "nucleus/io/fastq_reader.h"
#include "nucleus/platform/types.h"
#include "nucleus/protos/fastq.pb.h"
#include "nucleus/util/utils.h"
#include <utility> 

namespace tensorflow {

using nucleus::FastqReader;

class FastqOp : public OpKernel {
  public:
    explicit FastqOp(OpKernelConstruction* context) : OpKernel(context) {}
    ~FastqOp() {}

    void Compute(OpKernelContext* context) override {

      const Tensor& filename_tensor = context->input(0);
      const std::string& filename = filename_tensor.scalar<string>()();

      std::unique_ptr<FastqReader> reader = std::move(
        FastqReader::FromFile(filename,
                              nucleus::genomics::v1::FastqReaderOptions())
            .ValueOrDie());

      std::vector<std::string> nucleotides;

      std::shared_ptr<nucleus::FastqIterable> iterable = reader->Iterate().ValueOrDie();
      do {
        nucleus::genomics::v1::FastqRecord record;
        nucleus::StatusOr<bool> more = iterable->Next(&record);
        OP_REQUIRES(context, more.ok(), errors::Internal("internal error: ", more.error_message()));
        if (!more.ValueOrDie()) {
          break;
        }
        nucleotides.push_back(record.sequence());
      } while (true);

      TensorShape output_shape({static_cast<int64>(nucleotides.size())});
      Tensor* output_tensor;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

      for (size_t i = 0; i < nucleotides.size(); i++) {
        output_tensor->flat<string>()(i) = std::move(nucleotides[i]);
      }
    }
};

REGISTER_OP("FastqOp")
    .Input("filename: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });
REGISTER_KERNEL_BUILDER(Name("FastqOp").Device(DEVICE_CPU), FastqOp);

} // tensorflow
