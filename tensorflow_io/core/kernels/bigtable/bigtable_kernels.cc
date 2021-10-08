#include "tensorflow/core/framework/op_kernel.h"
#include "google/cloud/bigtable/table.h"

using namespace tensorflow;

namespace cbt = ::google::cloud::bigtable;

class ZeroOutOp : public OpKernel {
public:
    explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("project_id", &project_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("instance_id", &instance_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("columns", &columns_));
  }


    void Compute(OpKernelContext *context) override {


      std::cout << "got " << project_id_ << ":" << instance_id_ << ":" << table_id_ << "\n";
      std::cout << "columns len:" << columns_.size() << "\n";


      // Grab the input tensor

      cbt::Table table(cbt::CreateDefaultDataClient(project_id_, instance_id_,
                                                    cbt::ClientOptions()),
                       table_id_);

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
      OP_REQUIRES_OK(context, context->allocate_output(0, {5},
                                                       &output_tensor));
      auto output_flat = output_tensor->flat<string>();

      // Set all but the first element of the output tensor to 0.
      const int N = 5;
      for (int i = 0; i < N; i++) {
        output_flat(i) = "123";
      }
    }
private:
    string project_id_;
    string instance_id_;
    string table_id_;
    std::vector<string> columns_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableTest")
.
Device(DEVICE_CPU), ZeroOutOp
);
