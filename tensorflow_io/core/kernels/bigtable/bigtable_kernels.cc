#include "tensorflow/core/framework/op_kernel.h"
#include "google/cloud/bigtable/table.h"

using namespace tensorflow;

namespace cbt = ::google::cloud::bigtable;

class ZeroOutOp : public OpKernel {
public:
    explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
      // Grab the input tensor
      const Tensor &input_tensor = context->input(0);
      auto input = input_tensor.flat<int32>();

      cbt::Table table(cbt::CreateDefaultDataClient("test_project", "test_instance",
                                                    cbt::ClientOptions()),
                       "t1");

      google::cloud::bigtable::v1::RowReader reader1 = table.ReadRows(
              cbt::RowRange::InfiniteRange(), cbt::Filter::PassAllFilter());

      for (auto const &row: reader1) {
        if (!row) throw std::runtime_error(row.status().message());
        std::cout << " row: " << row->row_key() << ":\n";
        for (auto const &cell: row->cells()) {
          std::cout << "cell:\n";
          std::cout << cell.family_name() << ":" << cell.column_qualifier() << ":"
                    << cell.value() << "       @ " << cell.timestamp().count()
                    << "us\n";
        }
      }
      // Create an output tensor
      Tensor *output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                       &output_tensor));
      auto output_flat = output_tensor->flat<int32>();

      // Set all but the first element of the output tensor to 0.
      const int N = input.size();
      for (int i = 1; i < N; i++) {
        output_flat(i) = 0;
      }

      // Preserve the first input value if possible.
      if (N > 0) output_flat(0) = input(0);
    }
};

REGISTER_KERNEL_BUILDER(Name("BigtableTest")
.
Device(DEVICE_CPU), ZeroOutOp
);
