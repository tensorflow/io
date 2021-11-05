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
#include "absl/memory/memory.h"
#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_row_set.h"

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace data {
namespace {

tensorflow::error::Code GoogleCloudErrorCodeToTfErrorCode(::google::cloud::StatusCode code) {
  switch (code) {
    case ::google::cloud::StatusCode::kOk:
      return ::tensorflow::error::OK;
    case ::google::cloud::StatusCode::kCancelled:
      return ::tensorflow::error::CANCELLED;
    case ::google::cloud::StatusCode::kInvalidArgument:
      return ::tensorflow::error::INVALID_ARGUMENT;
    case ::google::cloud::StatusCode::kDeadlineExceeded:
      return ::tensorflow::error::DEADLINE_EXCEEDED;
    case ::google::cloud::StatusCode::kNotFound:
      return ::tensorflow::error::NOT_FOUND;
    case ::google::cloud::StatusCode::kAlreadyExists:
      return ::tensorflow::error::ALREADY_EXISTS;
    case ::google::cloud::StatusCode::kPermissionDenied:
      return ::tensorflow::error::PERMISSION_DENIED;
    case ::google::cloud::StatusCode::kUnauthenticated:
      return ::tensorflow::error::UNAUTHENTICATED;
    case ::google::cloud::StatusCode::kResourceExhausted:
      return ::tensorflow::error::RESOURCE_EXHAUSTED;
    case ::google::cloud::StatusCode::kFailedPrecondition:
      return ::tensorflow::error::FAILED_PRECONDITION;
    case ::google::cloud::StatusCode::kAborted:
      return ::tensorflow::error::ABORTED;
    case ::google::cloud::StatusCode::kOutOfRange:
      return ::tensorflow::error::OUT_OF_RANGE;
    case ::google::cloud::StatusCode::kUnimplemented:
      return ::tensorflow::error::UNIMPLEMENTED;
    case ::google::cloud::StatusCode::kInternal:
      return ::tensorflow::error::INTERNAL;
    case ::google::cloud::StatusCode::kUnavailable:
      return ::tensorflow::error::UNAVAILABLE;
    case ::google::cloud::StatusCode::kDataLoss:
      return ::tensorflow::error::DATA_LOSS;
    default:
      return ::tensorflow::error::UNKNOWN;
  }
}

Status GoogleCloudStatusToTfStatus(const ::google::cloud::Status& status) {
  if (status.ok()) {
    return Status::OK();
  }
  return Status(GoogleCloudErrorCodeToTfErrorCode(status.code()),
                strings::StrCat("Error reading from Cloud Bigtable: ",
                                status.message()));
}

class BigtableClientResource : public ResourceBase {
 public:
  explicit BigtableClientResource(const std::string& project_id,
                                  const std::string& instance_id)
      : data_client_(CreateDataClient(project_id, instance_id)) {}

  cbt::Table CreateTable(const std::string& table_id) {
    VLOG(1) << "CreateTable";
    return cbt::Table(data_client_, table_id);
  }

  string DebugString() const override { return "BigtableClientResource"; }

 private:
  std::shared_ptr<cbt::DataClient> CreateDataClient(
      const std::string& project_id, const std::string& instance_id) {
    VLOG(1) << "CreateDataClient";
    return cbt::CreateDefaultDataClient(project_id, instance_id,
                                        cbt::ClientOptions());
  }
  std::shared_ptr<cbt::DataClient> data_client_;
};

class BigtableClientOp : public OpKernel {
 public:
  explicit BigtableClientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("project_id", &project_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("instance_id", &instance_id_));
    VLOG(1) << "BigtableClientOp ctor";
  }

  void Compute(OpKernelContext* ctx) override TF_LOCKS_EXCLUDED(mu_) {
    VLOG(1) << "BigtableClientOp compute";
    ResourceMgr* mgr = ctx->resource_manager();
    ContainerInfo cinfo;
    OP_REQUIRES_OK(ctx, cinfo.Init(mgr, def()));

    BigtableClientResource* resource =
        new BigtableClientResource(project_id_, instance_id_);
    OP_REQUIRES_OK(ctx, mgr->Create<BigtableClientResource>(
                            cinfo.container(), cinfo.name(), resource));
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo.container(), cinfo.name(),
                            TypeIndex::Make<BigtableClientResource>()));
  }

 private:
  string project_id_;
  string instance_id_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableClient").Device(DEVICE_CPU),
                        BigtableClientOp);

template <typename Dataset>
class Iterator : public DatasetIterator<Dataset> {
 public:
  explicit Iterator(const typename DatasetIterator<Dataset>::Params& params,
                    const std::string& table_id,
                    const std::vector<std::string>& columns)
      : DatasetIterator<Dataset>(params),
        columns_(ColumnsToFamiliesAndQualifiers(columns)),
        reader_(
            this->dataset()->client_resource().CreateTable(table_id).ReadRows(
                cbt::RowRange::InfiniteRange(),
                cbt::Filter::Chain(CreateColumnsFilter(columns_),
                                   cbt::Filter::Latest(1)))),
        it_(this->reader_.begin()),
        column_to_idx_(CreateColumnToIdxMap(columns_)) {}

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    VLOG(1) << "GetNextInternal";
    mutex_lock l(mu_);
    if (it_ == reader_.end()) {
      VLOG(1) << "End of sequence";
      *end_of_sequence = true;
      return Status::OK();
    }
    *end_of_sequence = false;

    VLOG(1) << "alocating tensor";
    const std::size_t kNumCols = column_to_idx_.size();
    Tensor res(ctx->allocator({}), DT_STRING, {(long)kNumCols});
    auto res_data = res.tensor<tstring, 1>();

    VLOG(1) << "getting row";
    const auto& row = *it_;
    if (!row.ok()) {
      LOG(ERROR) << row.status().message();
      return GoogleCloudStatusToTfStatus(row.status());
    }
    for (const auto& cell : row.value().cells()) {
      std::pair<const std::string&, const std::string&> key(
          cell.family_name(), cell.column_qualifier());
      const auto column_idx = column_to_idx_.find(key);
      if (column_idx != column_to_idx_.end()) {
        VLOG(1) << "getting column:" << column_idx->second;
        res_data(column_idx->second) = std::move(cell.value());
      } else {
        LOG(ERROR) << "column " << cell.family_name() << ":"
                   << cell.column_qualifier()
                   << " was unexpectedly read from bigtable";
      }
    }
    VLOG(1) << "returning value";
    out_tensors->emplace_back(std::move(res));

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
  cbt::Filter CreateColumnsFilter(
      const std::vector<std::pair<std::string, std::string>>& columns) {
    VLOG(1) << "CreateColumnsFilter";
    std::vector<cbt::Filter> filters;

    for (const auto& column : columns) {
      cbt::Filter f = cbt::Filter::ColumnName(column.first, column.second);
      filters.push_back(std::move(f));
    }

    return cbt::Filter::InterleaveFromRange(filters.begin(), filters.end());
  }

  static std::pair<std::string, std::string> ColumnToFamilyAndQualifier(
      const std::string& col_name_full) {
    VLOG(1) << "ColumnToFamilyAndQualifier" << col_name_full;
    std::vector<std::string> result_vector = absl::StrSplit(col_name_full, ":");
    if (result_vector.size() != 2 || result_vector[0].empty())
      throw std::invalid_argument("Invalid column name:" + col_name_full +
                                  "\nColumn name must be in format " +
                                  "column_family:column_name.");
    return std::make_pair(result_vector[0], result_vector[1]);
  }

  static std::vector<std::pair<std::string, std::string>>
  ColumnsToFamiliesAndQualifiers(const std::vector<std::string>& columns) {
    VLOG(1) << "ColumnsToFamiliesAndQualifiers";
    std::vector<std::pair<std::string, std::string>> columnPairs(
        columns.size());
    std::transform(columns.begin(), columns.end(), columnPairs.begin(),
                   &ColumnToFamilyAndQualifier);
    return columnPairs;
  }

  static absl::flat_hash_map<std::pair<const std::string&, const std::string&>,
                             size_t>
  CreateColumnToIdxMap(
      const std::vector<std::pair<std::string, std::string>>& columns) {
    VLOG(1) << "CreateColumnToIdxMap";
    absl::flat_hash_map<std::pair<const std::string&, const std::string&>,
                        size_t>
        column_map;
    std::size_t index = 0;
    for (const auto& column : columns) {
      std::pair<const std::string&, const std::string&> key(column.first,
                                                            column.second);
      column_map[key] = index++;
    }
    return column_map;
  }

  mutex mu_;
  const std::shared_ptr<cbt::DataClient> data_client_;
  const std::vector<std::pair<std::string, std::string>> columns_;
  cbt::RowReader reader_ GUARDED_BY(mu_);
  cbt::v1::internal::RowReaderIterator it_ GUARDED_BY(mu_);
  // we're using a map with const refs to avoid copying strings when searching
  // for a value.
  const absl::flat_hash_map<std::pair<const std::string&, const std::string&>,
                            size_t>
      column_to_idx_;
};

class Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, BigtableClientResource* client_resource,
          std::string table_id, std::vector<std::string> columns)
      : DatasetBase(DatasetContext(ctx)),
        client_resource_(*client_resource),
        client_resource_unref_(client_resource),
        table_id_(table_id),
        columns_(columns) {
    dtypes_.push_back(DT_STRING);
    output_shapes_.push_back({});
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const override {
    VLOG(1) << "MakeIteratorInternal. table=" << table_id_;
    return absl::make_unique<Iterator<Dataset>>(
        typename DatasetIterator<Dataset>::Params{
            this, strings::StrCat(prefix, "::BigtableDataset")},
        table_id_, columns_);
  }

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  std::string DebugString() const override {
    return "BigtableDatasetOp::Dataset";
  }

  BigtableClientResource& client_resource() const { return client_resource_; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    return errors::Unimplemented("%s does not support serialization",
                                 DebugString());
  }

  Status CheckExternalState() const override { return Status::OK(); }

 private:
  BigtableClientResource& client_resource_;
  const core::ScopedUnref client_resource_unref_;
  const std::string table_id_;
  const std::vector<std::string> columns_;
  DataTypeVector dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

class BigtableDatasetOp : public DatasetOpKernel {
 public:
  explicit BigtableDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("columns", &columns_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    VLOG(1) << "Make Dataset";
    BigtableClientResource* client_resource;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &client_resource));
    core::ScopedUnref client_resource_unref_(client_resource);
    *output = new Dataset(ctx, client_resource, table_id_, columns_);
  }

 private:
  std::string table_id_;
  std::vector<std::string> columns_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableDataset").Device(DEVICE_CPU),
                        BigtableDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
