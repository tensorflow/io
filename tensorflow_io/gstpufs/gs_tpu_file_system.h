#ifndef THIRD_PARTY_GSTPUFS_GS_TPU_FILE_SYSTEM_H_
#define THIRD_PARTY_GSTPUFS_GS_TPU_FILE_SYSTEM_H_

#include "tensorflow/core/platform/cloud/gcs_file_system.h"
#include "absl/memory/memory.h"

namespace tensorflow {

// GsTpuFileSystem is implemented simply to register "gstpu://" as a file system
// scheme. It is a class of TpuGcsFileSystem, which implements all of the logic.
class GsTpuFileSystem : public GcsFileSystem {
 public:
  GsTpuFileSystem();

 protected:
  /// \brief Splits a GCS path to a bucket and an object.
  ///
  /// For example, "gstpu://bucket-name/path/to/file.txt" gets split into
  /// "bucket-name" and "path/to/file.txt".
  /// If fname only contains the bucket and empty_object_ok = true, the returned
  /// object is empty.
  Status ParseGcsPath(StringPiece fname, bool empty_object_ok, string* bucket,
                      string* object) override;
};

/// Google Cloud Storage implementation of a file system with retry on failures.
class RetryingGsTpuFileSystem : public RetryingFileSystem<GsTpuFileSystem> {
 public:
  RetryingGsTpuFileSystem()
      : RetryingFileSystem(absl::make_unique<GsTpuFileSystem>(),
                           RetryConfig(100000 /* init_delay_time_us */)) {}
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_GSTPUFS_GS_TPU_FILE_SYSTEM_H_
