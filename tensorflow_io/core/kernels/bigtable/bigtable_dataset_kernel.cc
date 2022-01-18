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
#include "google/cloud/bigtable/row_set.h"
#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_row_set.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_version_filters.h"
#include "tensorflow_io/core/kernels/bigtable/serialization.h"

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace data {
namespace {

tensorflow::error::Code GoogleCloudErrorCodeToTfErrorCode(
    ::google::cloud::StatusCode code) {
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
  return Status(
      GoogleCloudErrorCodeToTfErrorCode(status.code()),
      strings::StrCat("Error reading from Cloud Bigtable: ", status.message()));
}

class BigtableClientResource : public ResourceBase {
 public:
  explicit BigtableClientResource(const std::string& project_id,
                                  const std::string& instance_id)
      : data_client_(CreateDataClient(project_id, instance_id)) {
    VLOG(1) << "BigtableClientResource ctor";
  }

  const std::shared_ptr<cbt::DataClient>& data_client() const {
    return data_client_;
  }

  ~BigtableClientResource() { VLOG(1) << "BigtableClientResource dtor"; }

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

  ~BigtableClientOp() { VLOG(1) << "BigtableClientOp dtor"; }

  void Compute(OpKernelContext* ctx) override {
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
                    const std::vector<std::string>& columns)
      : DatasetIterator<Dataset>(params),
        columns_(ColumnsToFamiliesAndQualifiers(columns)),
        reader_(this->dataset()->CreateTable().ReadRows(
            this->dataset()->row_set(),
            cbt::Filter::Chain(CreateColumnsFilter(columns_),
                               this->dataset()->filter(),
                               cbt::Filter::Latest(1)))),
        it_(this->reader_.begin()),
        column_to_idx_(CreateColumnToIdxMap(columns_)) {
    VLOG(1) << "DatasetIterator ctor";
  }

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
    const DataType dtype = this->dataset()->output_type();
    Tensor res(ctx->allocator({}), dtype, {(long)kNumCols});

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
        TF_RETURN_IF_ERROR(
            io::PutCellValueInTensor(res, column_idx->second, dtype, cell));
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

    return filters.size() > 1 ? cbt::Filter::InterleaveFromRange(
                                    filters.begin(), filters.end())
                              : filters[0];
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
  Dataset(OpKernelContext* ctx,
          const std::shared_ptr<cbt::DataClient>& data_client,
          cbt::RowSet row_set, cbt::Filter filter, std::string table_id,
          std::vector<std::string> columns, DataType output_type)
      : DatasetBase(DatasetContext(ctx)),
        data_client_(data_client),
        row_set_(std::move(row_set)),
        filter_(std::move(filter)),
        output_type_(std::move(output_type)),
        table_id_(table_id),
        columns_(columns) {
    dtypes_.push_back({output_type_});
    output_shapes_.push_back({});
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const override {
    VLOG(1) << "MakeIteratorInternal. table=" << table_id_;
    return absl::make_unique<Iterator<Dataset>>(
        typename DatasetIterator<Dataset>::Params{
            this, strings::StrCat(prefix, "::BigtableDataset")},
        columns_);
  }

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  const DataType output_type() const { return output_type_; }

  std::string DebugString() const override {
    return "BigtableDatasetOp::Dataset";
  }

  const std::shared_ptr<cbt::DataClient>& data_client() const {
    return data_client_;
  }
  const cbt::RowSet& row_set() const { return row_set_; }

  cbt::Table CreateTable() const {
    VLOG(1) << "CreateTable";
    cbt::Table table(data_client_, table_id_);
    VLOG(1) << "table crated";
    return table;
  }

  const cbt::Filter& filter() const { return filter_; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    return errors::Unimplemented("%s does not support serialization",
                                 DebugString());
  }

  Status CheckExternalState() const override { return Status::OK(); }

 private:
  std::shared_ptr<cbt::DataClient> const& data_client_;
  const cbt::RowSet row_set_;
  cbt::Filter filter_;
  DataType output_type_;
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_type", &output_type_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    VLOG(1) << "Make Dataset";
    BigtableClientResource* client_resource;
    OP_REQUIRES_OK(ctx,
                   GetResourceFromContext(ctx, "client", &client_resource));
    core::ScopedUnref unref_client(client_resource);

    io::BigtableRowSetResource* row_set_resource;
    OP_REQUIRES_OK(ctx,
                   GetResourceFromContext(ctx, "row_set", &row_set_resource));
    core::ScopedUnref row_set_resource_unref_(row_set_resource);

    io::BigtableFilterResource* filter_resource;
    OP_REQUIRES_OK(ctx,
                   GetResourceFromContext(ctx, "filter", &filter_resource));
    core::ScopedUnref filter_resource_unref_(filter_resource);

    *output = new Dataset(
        ctx, client_resource->data_client(), row_set_resource->row_set(),
        filter_resource->filter(), table_id_, columns_, output_type_);
  }

 private:
  std::string table_id_;
  std::vector<std::string> columns_;
  DataType output_type_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableDataset").Device(DEVICE_CPU),
                        BigtableDatasetOp);

// Return the index of the tablet that a worker should start with. Each worker
// start with their first tablet and finish on tablet before next worker's first
// tablet. Each worker should get num_tablets/num_workers rounded down, plus at
// most one. If we simply round up, then the last worker may be starved.
// Consider an example where there's 100 tablets and 11 workers. If we give
// round_up(100/11) to each one, then first 10 workers get 10 tablets each, and
// the last one gets nothing.
int GetWorkerStartIndex(size_t num_tablets, size_t num_workers,
                        size_t worker_id) {
  // if there's more workers than tablets, workers get one tablet each or less.
  if (num_tablets <= num_workers) return std::min(num_tablets, worker_id);
  // tablets_per_worker: minimum tablets each worker should obtain.
  size_t const tablets_per_worker = num_tablets / num_workers;
  // surplus_tablets: excess that has to be evenly distributed among the workers
  // so that no worker gets more than tablets_per_worker + 1.
  size_t const surplus_tablets = num_tablets % num_workers;
  size_t const workers_before = worker_id;
  return tablets_per_worker * workers_before +
         std::min(surplus_tablets, workers_before);
}

bool RowSetIntersectsRange(cbt::RowSet const& row_set,
                           std::string const& start_key,
                           std::string const& end_key) {
  auto range = cbt::RowRange::Range(start_key, end_key);
  return !row_set.Intersect(range).IsEmpty();
}

class BigtableSplitRowSetEvenlyOp : public OpKernel {
 public:
  explicit BigtableSplitRowSetEvenlyOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    VLOG(1) << "BigtableSplitRowSetEvenlyOp ctor ";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_splits", &num_splits_));
  }

  void Compute(OpKernelContext* context) override {
    mutex_lock l(mu_);

    ResourceMgr* mgr = context->resource_manager();
    ContainerInfo cinfo;
    OP_REQUIRES_OK(context, cinfo.Init(mgr, def()));

    BigtableClientResource* client_resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "client", &client_resource));
    core::ScopedUnref unref_client(client_resource);

    io::BigtableRowSetResource* row_set_resource;
    OP_REQUIRES_OK(
        context, GetResourceFromContext(context, "row_set", &row_set_resource));
    core::ScopedUnref unref_row_set(row_set_resource);

    VLOG(1) << "BigtableSplitRowSetEvenlyOp got RowSet: "
            << row_set_resource->ToString();
    if (row_set_resource->row_set().IsEmpty()) {
      OP_REQUIRES_OK(context,
                     errors::FailedPrecondition("row_set cannot be empty!"));
    }

    auto table = cbt::Table(client_resource->data_client(), table_id_);
    auto maybe_sample_row_keys = table.SampleRows();
    OP_REQUIRES_OK(context,
                   GoogleCloudStatusToTfStatus(maybe_sample_row_keys.status()));

    auto& sample_row_keys = maybe_sample_row_keys.value();

    std::vector<std::pair<std::string, std::string>> tablets;

    std::string start_key;
    for (auto& sample_row_key : sample_row_keys) {
      auto& end_key = sample_row_key.row_key;
      tablets.emplace_back(start_key, end_key);
      start_key = std::move(end_key);
    }
    if (!start_key.empty() || tablets.size() == 0) {
      tablets.emplace_back(start_key, "");
    }
    tablets.erase(
        std::remove_if(
            tablets.begin(), tablets.end(),
            [row_set_resource](std::pair<std::string, std::string> const& p) {
              return !RowSetIntersectsRange(row_set_resource->row_set(),
                                            p.first, p.second);
            }),
        tablets.end());

    VLOG(1) << "got array of tablets of size:" << tablets.size();

    size_t output_size = std::min<std::size_t>(tablets.size(), num_splits_);

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {static_cast<long>(output_size)},
                                            &output_tensor));
    auto output_v = output_tensor->tensor<ResourceHandle, 1>();

    for (size_t i = 0; i < output_size; i++) {
      size_t start_idx = GetWorkerStartIndex(tablets.size(), output_size, i);
      size_t next_worker_start_idx =
          GetWorkerStartIndex(tablets.size(), output_size, i + 1);
      size_t end_idx = next_worker_start_idx - 1;
      start_key = tablets.at(start_idx).first;
      std::string end_key = tablets.at(end_idx).second;
      io::BigtableRowSetResource* work_chunk_row_set =
          new io::BigtableRowSetResource(row_set_resource->Intersect(
              cbt::RowRange::RightOpen(start_key, end_key)));

      std::string container_name = cinfo.name() + std::to_string(i);

      VLOG(1) << "creating resource:" << cinfo.container() << ":"
              << container_name;

      OP_REQUIRES_OK(
          context, mgr->Create<io::BigtableRowSetResource>(
                       cinfo.container(), container_name, work_chunk_row_set));
      output_v(i) = MakeResourceHandle(
          cinfo.container(), container_name, *context->device(),
          TypeIndex::Make<io::BigtableRowSetResource>());
    }
  }

 private:
  mutex mu_;
  std::string table_id_ GUARDED_BY(mu_);
  int num_splits_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("BigtableSplitRowSetEvenly").Device(DEVICE_CPU),
                        BigtableSplitRowSetEvenlyOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
