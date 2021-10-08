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


      std::map<std::string, int> column_map;
      int counter = 0;
      for(auto const& column : columns_){
        column_map[column] = counter++;
      }

      // Grab the input tensor

      cbt::Table table(cbt::CreateDefaultDataClient(project_id_, instance_id_,
                                                    cbt::ClientOptions()),
                       table_id_);

      google::cloud::bigtable::v1::RowReader reader1 = table.ReadRows(
              cbt::RowRange::InfiniteRange(), cbt::Filter::PassAllFilter());

      std::vector<std::vector<std::string>> rows_vec;


      for (auto const &row: reader1) {
        if (!row) throw std::runtime_error(row.status().message());
        std::vector<std::string> row_vec(column_map.size());
        std::fill(row_vec.begin(), row_vec.end(), "Nothing");

        std::cout << " row: " << row->row_key() << ":\n";
        for (auto const &cell: row->cells()) {
          std::cout << "cell:\n";
          std::cout << cell.family_name() << ":" << cell.column_qualifier() << ":"
                    << cell.value() << "       @ " << cell.timestamp().count()
                    << "us\n";
          std::string col_name = cell.family_name() + ":" + cell.column_qualifier();
          std::cout << "col_name:" << col_name << "\n";
          if(column_map.find(col_name) != column_map.end()){
            std::cout << "found!\n";
            row_vec[column_map[col_name]] = cell.value();
          }
        }
        rows_vec.push_back(row_vec);
      }

      // Create an output tensor
      Tensor *output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, {rows_vec.size(), column_map.size()},
                                                       &output_tensor));
      auto output_v = output_tensor->tensor<tstring, 2>();

      // Set all but the first element of the output tensor to 0.
      const int N_rows = output_tensor->shape().dim_size(0);
      const int N_cols = output_tensor->shape().dim_size(1);
      std::cout<< "N_rows:" << N_rows << "\n";
      std::cout<< "N_cols:" << N_cols << "\n";

      for (int i = 0; i < N_rows; i++) {
        for(int j=0; j<N_cols; j++){
          output_v(i, j) = rows_vec[i][j];
        }
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
