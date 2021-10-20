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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using ::tensorflow::DT_STRING;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace data {
namespace {


template <typename Dataset>
class Iterator : public DatasetIterator<Dataset> {
 public:
  explicit Iterator(const typename DatasetIterator<Dataset>::Params& params, std::string const& project_id,
                    std::string const& instance_id, std::string const& table_id,
                    std::vector<std::string> columns)
      : DatasetIterator<Dataset>(params),
        data_client_(CreateDataClient(project_id, instance_id)),
        reader_(
            CreateTable(this->data_client_, table_id)
                ->ReadRows(cbt::RowRange::InfiniteRange(),
                           cbt::Filter::Chain(CreateColumnsFilter(column_to_idx_),
                                              cbt::Filter::Latest(1)))),
        it_(this->reader_.begin()),
        column_to_idx_(CreateColumnMap(columns)) {}

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    VLOG(1) << "GetNextInternal";
    mutex_lock l(mu_);
    if (it_ == reader_.end()) {
      VLOG(1) << "End of sequence";
      *end_of_sequence = true;
      return Status::OK();
    }

    VLOG(1) << "alocating tensor";
    long n_cols = column_to_idx_.size();
    Tensor res(ctx->allocator({}), DT_STRING, {n_cols});
    auto res_data = res.tensor<tstring, 1>();

    VLOG(1) << "getting row";
    auto const& row = *it_;
    for (const auto& cell : row.value().cells()) {
      auto const key = std::make_pair(cell.family_name(),
                                              cell.column_qualifier());
      VLOG(1) << "getting column:" << column_to_idx_[key];
      res_data(column_to_idx_[key]) = std::move(cell.value());
    }
    VLOG(1) << "returning value";
    out_tensors->emplace_back(std::move(res));
    *end_of_sequence = false;

    VLOG(1) << "incrementing iterator";
    it_ = std::next(it_);

    return Status::OK();
  }

 protected:
  Status SaveInternal(SerializationContext* ctx,
                      IteratorStateWriter* writer) override {
    return errors::Unimplemented("SaveInternal");
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    return errors::Unimplemented(
        "Iterator does not support 'RestoreInternal')");
  }

 private:
  std::unique_ptr<cbt::Table> CreateTable(
      std::shared_ptr<cbt::DataClient> const& data_client,
      std::string const& table_id) {
    VLOG(1) << "CreateTable";
    return std::make_unique<cbt::Table>(data_client, table_id);
  }

  std::shared_ptr<cbt::DataClient> CreateDataClient(
      std::string const& project_id, std::string const& instance_id) {
    VLOG(1) << "CreateDataClient";
    return cbt::CreateDefaultDataClient(
        std::move(project_id), std::move(instance_id), cbt::ClientOptions());
  }

  cbt::Filter CreateColumnsFilter(
      absl::flat_hash_map<std::pair<std::string, std::string>, size_t> const& columns) {
    VLOG(1) << "CreateColumnsFilter";
    std::vector<cbt::Filter> filters;

    for (const auto& key : columns) {
      std::pair<std::string, std::string> column = key.first;
      cbt::Filter f = cbt::Filter::ColumnName(std::move(column.first), std::move(column.second));
      filters.push_back(std::move(f));
    }

    return cbt::Filter::InterleaveFromRange(filters.begin(), filters.end());
  }

  static std::pair<std::string, std::string> ColumnNameToPair(
      std::string const& col_name_full) {
    size_t delimiter_pos = col_name_full.find(':');
    if (delimiter_pos == std::string::npos)
      throw std::invalid_argument("Invalid column name:" + col_name_full +
                                  "\nColumn name must be in format " +
                                  "column_family:column_name.");
    std::string col_family = col_name_full.substr(0, delimiter_pos);
    std::string col_name =
        col_name_full.substr(delimiter_pos + 1, col_name_full.length());
    std::pair<std::string, std::string> pair(col_family, col_name);
    return pair;
  }

  static absl::flat_hash_map<std::pair<std::string, std::string>, size_t> CreateColumnMap(
      std::vector<std::string> const& columns) {
    absl::flat_hash_map<std::pair<std::string, std::string>, size_t> column_map;
    size_t index = 0;
    for (const auto& column_name : columns) {
      std::pair<std::string, std::string> pair = ColumnNameToPair(column_name);
      column_map[pair] = index++;
    }
    return column_map;
  }

  mutex mu_;
  std::shared_ptr<cbt::DataClient> data_client_ GUARDED_BY(mu_);
  absl::flat_hash_map<std::pair<std::string, std::string>, size_t> column_to_idx_ GUARDED_BY(mu_);
  cbt::RowReader reader_ GUARDED_BY(mu_);
  cbt::v1::internal::RowReaderIterator it_ GUARDED_BY(mu_);
};


class Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, std::string project_id, std::string instance_id,
          std::string table_id, std::vector<std::string> columns)
      : DatasetBase(DatasetContext(ctx)),
        project_id_(project_id),
        instance_id_(instance_id),
        table_id_(table_id),
        columns_(columns) {
    size_t num_outputs = columns_.size();
    dtypes_.push_back(DT_STRING);
    output_shapes_.push_back({});
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const {
    VLOG(1) << "MakeIteratorInternal. table=" << project_id_ << ":"
            << instance_id_ << ":" << table_id_;
    return  std::unique_ptr<IteratorBase>(
        new Iterator<Dataset>({this, strings::StrCat(prefix, "::BigtableDataset")},
                     project_id_, instance_id_, table_id_, columns_));
  }

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  std::string DebugString() const override {
    return "BigtableDatasetOp::Dataset";
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b, Node** output) const {
    return errors::Unimplemented("%s does not support serialization",
                                 DebugString());
  }

  Status CheckExternalState() const override { return Status::OK(); }

 private:
  std::string project_id_;
  std::string instance_id_;
  std::string table_id_;
  std::vector<std::string> columns_;
  DataTypeVector dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};


class BigtableDatasetOp : public DatasetOpKernel {
 public:
  explicit BigtableDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("project_id", &project_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("instance_id", &instance_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("columns", &columns_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    VLOG(1) << "Make Dataset";

    *output = new Dataset(ctx, project_id_, instance_id_, table_id_, columns_);
  }

 private:
  std::string project_id_;
  std::string instance_id_;
  std::string table_id_;
  std::vector<std::string> columns_;
};



REGISTER_KERNEL_BUILDER(Name("BigtableDataset").Device(DEVICE_CPU),
                        BigtableDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
