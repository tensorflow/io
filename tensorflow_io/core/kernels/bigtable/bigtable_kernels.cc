#include "tensorflow/core/framework/op_kernel.h"
#include "google/cloud/bigtable/table.h"

using namespace tensorflow;

namespace cbt = ::google::cloud::bigtable;

class BigtableTestOp : public OpKernel {
public:
    explicit BigtableTestOp(OpKernelConstruction *context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("project_id", &project_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("instance_id", &instance_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("columns", &columns_));

      int index = 0;
      for(auto const& column : columns_){
        column_map[column] = index++;
      }

  }


    void Compute(OpKernelContext *context) override {
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

        for (auto const &cell: row->cells()) {
          std::string col_name = cell.family_name() + ":" + cell.column_qualifier();
          if(column_map.find(col_name) != column_map.end()){
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
    std::map<std::string, int> column_map;
};

REGISTER_KERNEL_BUILDER(Name("BigtableTest").Device(DEVICE_CPU), BigtableTestOp);
