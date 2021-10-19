/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "google/cloud/bigtable/table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

using namespace tensorflow;

namespace cbt = ::google::cloud::bigtable;

class BigtableTestOp : public OpKernel {
 public:
  explicit BigtableTestOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context, context->GetAttr("project_id", &project_id_));
    OP_REQUIRES_OK(context, context->GetAttr("instance_id", &instance_id_));
    OP_REQUIRES_OK(context, context->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(context, context->GetAttr("columns", &columns_));

    int index = 0;
    for (auto const& column : columns_) {
      column_map[column] = index++;
    }
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor

    cbt::Table table(cbt::CreateDefaultDataClient(project_id_, instance_id_,
                                                  cbt::ClientOptions()),
                     table_id_);

    google::cloud::bigtable::v1::RowReader reader1 = table.ReadRows(
        cbt::RowRange::InfiniteRange(), cbt::Filter::PassAllFilter());

    std::vector<std::vector<std::string>> rows_vec;

    for (auto const& row : reader1) {
      if (!row) throw std::runtime_error(row.status().message());
      std::vector<std::string> row_vec(column_map.size());
      std::fill(row_vec.begin(), row_vec.end(), "Nothing");

      for (auto const& cell : row->cells()) {
        std::string col_name =
            cell.family_name() + ":" + cell.column_qualifier();
        if (column_map.find(col_name) != column_map.end()) {
          row_vec[column_map[col_name]] = cell.value();
        }
      }
      rows_vec.push_back(row_vec);
    }

    // Create an output tensor
    Tensor* output_tensor = NULL;
    long N_rows = (long)rows_vec.size();
    long N_cols = (long)column_map.size();
    OP_REQUIRES_OK(
        context, context->allocate_output(0, {N_rows, N_cols}, &output_tensor));
    auto output_v = output_tensor->tensor<tstring, 2>();

    // Set all but the first element of the output tensor to 0.

    for (int i = 0; i < N_rows; i++) {
      for (int j = 0; j < N_cols; j++) {
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

REGISTER_KERNEL_BUILDER(Name("BigtableTest").Device(DEVICE_CPU),
                        BigtableTestOp);
