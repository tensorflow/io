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

class BigtableDatasetOp : public DatasetOpKernel {
 public:
  explicit BigtableDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("project_id", &project_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("instance_id", &instance_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("columns", &columns_));
  }

  void MakeDataset(OpKernelContext* ctx,
                   DatasetBase** output) override {
    // Parse and validate any input tensors that define the dataset using
    // `ctx->input()` or the utility function
    // `ParseScalarArgument<T>(ctx, &arg)`.
    VLOG(0) << "Make Dataset";

    // Create the dataset object, passing any (already-validated) arguments from
    // attrs or input tensors.
    *output = new Dataset(ctx, project_id_, instance_id_, table_id_, columns_);
  }

 private:
  std::string project_id_;
  std::string instance_id_;
  std::string table_id_;
  std::vector<std::string> columns_;

  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::string project_id,
            std::string instance_id, std::string table_id,
            std::vector<std::string> columns)
        : DatasetBase(DatasetContext(ctx)),
          project_id_(project_id),
          instance_id_(instance_id),
          table_id_(table_id),
          columns_(columns) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const std::string& prefix) const {
      VLOG(0) << "MakeIteratorInternal. instance=" << project_id_ << ":"
              << instance_id_ << ":" << table_id_;
      return std::unique_ptr<IteratorBase>(new Iterator(
          {this, strings::StrCat(prefix, "::BigtableDataset")},
          project_id_, instance_id_, table_id_, columns_.size()));
    }

    // Record structure: Each record is represented by a scalar string tensor.
    //
    // Dataset elements can have a fixed number of components of different
    // types and shapes; replace the following two methods to customize this
    // aspect of the dataset.
    const DataTypeVector& output_dtypes() const override {
      static auto* const dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    std::string DebugString() const override {
      return "BigtableDatasetOp::Dataset";
    }

   protected:
    // Optional: Implementation of `GraphDef` serialization for this dataset.
    //
    // Implement this method if you want to be able to save and restore
    // instances of this dataset (and any iterators over it).
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const {
      // Construct nodes to represent any of the input tensors from this
      // object's member variables using `b->AddScalar()` and `b->AddVector()`.

      return errors::Unimplemented(
          "%s does not support serialization", DebugString());
    }

    Status CheckExternalState() const override { return Status::OK(); }

   private:
    std::string project_id_;
    std::string instance_id_;
    std::string table_id_;
    std::vector<std::string> columns_;

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params, std::string const& project_id,
                        std::string const& instance_id,
                        std::string const& table_id, int num_cols)
          : DatasetIterator<Dataset>(params),
            data_client_(CreateDataClient(project_id, instance_id)),
            reader_(CreateTable(this->data_client_, table_id)
                        ->ReadRows(cbt::RowRange::InfiniteRange(),
                                   cbt::Filter::PassAllFilter())),
            it_(this->reader_.begin()),
            num_cols(num_cols) {}

      // Implementation of the reading logic.
      //
      // The example implementation in this file yields the string "MyReader!"
      // ten times. In general there are three cases:
      //
      // 1. If an element is successfully read, store it as one or more tensors
      //    in `*out_tensors`, set `*end_of_sequence = false` and return
      //    `Status::OK()`.
      // 2. If the end of input is reached, set `*end_of_sequence = true` and
      //    return `Status::OK()`.
      // 3. If an error occurs, return an error status using one of the helper
      //    functions from "tensorflow/core/lib/core/errors.h".
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // NOTE: `GetNextInternal()` may be called concurrently, so it is
        // recommended that you protect the iterator state with a mutex.

        VLOG(0) << "GetNextInternal";
        mutex_lock l(mu_);
        if (it_ == reader_.end()) {
          VLOG(0) << "End of sequence";
          *end_of_sequence = true;
        } else {
          VLOG(0) << "alocating tensor";
          Tensor record_tensor(ctx->allocator({}), DT_STRING,
                                           {num_cols});
          auto record_v = record_tensor.tensor<tstring, 1>();

          VLOG(0) << "getting row";
          auto const& row = *it_;
          int counter = 0;
          for (const auto& cell : row.value().cells()) {
            VLOG(0) << "getting column:" << counter;
            record_v(counter++) = cell.value();
          }
          VLOG(0) << "returning value";
          out_tensors->emplace_back(std::move(record_tensor));
          *end_of_sequence = false;

          VLOG(0) << "incrementing iterator";
          it_ = std::next(it_);
        }

        return Status::OK();
      }

     protected:
      Status SaveInternal(
          SerializationContext* ctx,
          IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal");
      }

      Status RestoreInternal(
          IteratorContext* ctx,
          IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "Iterator does not support 'RestoreInternal')");
      }

     private:
      std::unique_ptr<cbt::Table> CreateTable(
          std::shared_ptr<cbt::DataClient> const& data_client,
          std::string const& table_id) {
        VLOG(0) << "CreateTable";
        return std::make_unique<cbt::Table>(data_client, table_id);
      }

      std::shared_ptr<cbt::DataClient> CreateDataClient(
          std::string const& project_id, std::string const& instance_id) {
        VLOG(0) << "CreateDataClient";
        return cbt::CreateDefaultDataClient(std::move(project_id),
                                            std::move(instance_id),
                                            cbt::ClientOptions());
      }

      cbt::Filter CreateColumnsFilter(
          std::map<std::pair<std::string, std::string>, size_t> const&
              columns) {
        VLOG(0) << "CreateColumnsFilter";
        std::vector<cbt::Filter> filters;

        for (const auto& key : columns) {
          std::pair<std::string, std::string> pair = key.first;
          cbt::Filter f = cbt::Filter::ColumnName(pair.first, pair.second);
          filters.push_back(std::move(f));
        }

        return cbt::Filter::InterleaveFromRange(filters.begin(), filters.end());
      }

      std::pair<std::string, std::string> ColumnNameToPair(
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

      mutex mu_;
      std::shared_ptr<cbt::DataClient> data_client_ GUARDED_BY(mu_);
      cbt::RowReader reader_ GUARDED_BY(mu_);
      cbt::v1::internal::RowReaderIterator it_ GUARDED_BY(mu_);
      int num_cols;
    };
  };
};

// Register the kernel implementation for MyReaderDataset.
REGISTER_KERNEL_BUILDER(Name("BigtableDataset").Device(DEVICE_CPU),
                        BigtableDatasetOp);


}  // namespace
}  // namespace data
}  // namespace tensorflow