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

#ifndef BIGTABLE_ROW_RANGE_H
#define BIGTABLE_ROW_RANGE_H

#include "google/cloud/bigtable/table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_resource_kernel.h"

namespace tensorflow {
namespace io {

class BigtableRowRangeResource : public ResourceBase {
 public:
  explicit BigtableRowRangeResource(google::cloud::bigtable::RowRange row_range)
      : row_range_(std::move(row_range)) {
    VLOG(1) << "BigtableRowRangeResource ctor";
  }

  ~BigtableRowRangeResource() { VLOG(1) << "BigtableRowRangeResource dtor"; }

  std::string ToString() const {
    std::string res;
    google::protobuf::TextFormat::PrintToString(row_range_.as_proto(), &res);
    return res;
  }

  google::cloud::bigtable::RowRange& row_range() { return row_range_; }

  string DebugString() const override {
    return "BigtableRowRangeResource:{" + ToString() + "}";
  }

 private:
  google::cloud::bigtable::RowRange row_range_;
};

}  // namespace io
}  // namespace tensorflow

#endif /* BIGTABLE_ROW_RANGE_H */
