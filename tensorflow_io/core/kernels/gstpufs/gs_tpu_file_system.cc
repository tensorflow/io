#include "tensorflow_io/core/kernels/gstpufs/gs_tpu_file_system.h"

namespace tensorflow {

GsTpuFileSystem::GsTpuFileSystem() : GcsFileSystem() {
  VLOG(1) << "Entering GsTpuFileSystem::GsTpuFileSystem";
}

Status GsTpuFileSystem::ParseGcsPath(StringPiece fname, bool empty_object_ok,
                                     string* bucket, string* object) {
  return ParseGcsPathForScheme(fname, "gstpu", empty_object_ok, bucket, object);
}

}  // namespace tensorflow

REGISTER_FILE_SYSTEM("gstpu", ::tensorflow::RetryingGsTpuFileSystem);
