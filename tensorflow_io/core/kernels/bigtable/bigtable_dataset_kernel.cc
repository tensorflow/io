#include "google/cloud/bigtable/table.h"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using ::tensorflow::DT_STRING;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;

namespace cbt = ::google::cloud::bigtable;

class BigtableDatasetOp : public tensorflow::data::DatasetOpKernel {
 public:
  explicit BigtableDatasetOp(tensorflow::OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("project_id", &project_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("instance_id", &instance_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("columns", &columns_));
  }

  void MakeDataset(tensorflow::OpKernelContext* ctx,
                   tensorflow::data::DatasetBase** output) override {
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

  class Dataset : public tensorflow::data::DatasetBase {
   public:
    Dataset(tensorflow::OpKernelContext* ctx, std::string project_id,
            std::string instance_id, std::string table_id,
            std::vector<std::string> columns)
        : tensorflow::data::DatasetBase(tensorflow::data::DatasetContext(ctx)),
          project_id_(project_id),
          instance_id_(instance_id),
          table_id_(table_id),
          columns_(columns) {}

    std::unique_ptr<tensorflow::data::IteratorBase> MakeIteratorInternal(
        const std::string& prefix) const {
    VLOG(0) << "MakeIteratorInternal. instance=" << project_id_ << ":" << instance_id_ << ":" << table_id_ ;
      return std::unique_ptr<tensorflow::data::IteratorBase>(new Iterator(
          {this, tensorflow::strings::StrCat(prefix, "::BigtableDataset")},
          project_id_, instance_id_, table_id_, columns_.size()));
    }

    // Record structure: Each record is represented by a scalar string tensor.
    //
    // Dataset elements can have a fixed number of components of different
    // types and shapes; replace the following two methods to customize this
    // aspect of the dataset.
    const tensorflow::DataTypeVector& output_dtypes() const override {
      static auto* const dtypes = new tensorflow::DataTypeVector({DT_STRING});
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
    Status AsGraphDefInternal(tensorflow::data::SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              tensorflow::Node** output) const {
      // Construct nodes to represent any of the input tensors from this
      // object's member variables using `b->AddScalar()` and `b->AddVector()`.

      return tensorflow::errors::Unimplemented(
          "%s does not support serialization", DebugString());
    }

    Status CheckExternalState() const override { return Status::OK(); }

   private:
    std::string project_id_;
    std::string instance_id_;
    std::string table_id_;
    std::vector<std::string> columns_;

    class Iterator : public tensorflow::data::DatasetIterator<Dataset> {
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
      Status GetNextInternal(tensorflow::data::IteratorContext* ctx,
                             std::vector<tensorflow::Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // NOTE: `GetNextInternal()` may be called concurrently, so it is
        // recommended that you protect the iterator state with a mutex.

            VLOG(0) << "GetNextInternal";
        tensorflow::mutex_lock l(mu_);
        if (it_ == reader_.end()) {
            VLOG(0) << "End of sequence";
          *end_of_sequence = true;
        } else {

            VLOG(0) << "alocating tensor";
          tensorflow::Tensor record_tensor(ctx->allocator({}), DT_STRING,
                                           {num_cols});
          auto record_v = record_tensor.tensor<tensorflow::tstring, 1>();

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
          tensorflow::data::SerializationContext* ctx,
          tensorflow::data::IteratorStateWriter* writer) override {
        return tensorflow::errors::Unimplemented("SaveInternal");
      }

      Status RestoreInternal(
          tensorflow::data::IteratorContext* ctx,
          tensorflow::data::IteratorStateReader* reader) override {
        return tensorflow::errors::Unimplemented(
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
        return cbt::CreateDefaultDataClient(
            std::move(project_id), std::move(instance_id),
            cbt::ClientOptions());
      }

      tensorflow::mutex mu_;
      std::shared_ptr<cbt::DataClient> data_client_ GUARDED_BY(mu_);
      cbt::RowReader reader_ GUARDED_BY(mu_);
      cbt::v1::internal::RowReaderIterator it_ GUARDED_BY(mu_);
      int num_cols;
    };
  };
};

// Register the kernel implementation for MyReaderDataset.
REGISTER_KERNEL_BUILDER(Name("BigtableDataset").Device(tensorflow::DEVICE_CPU),
                        BigtableDatasetOp);