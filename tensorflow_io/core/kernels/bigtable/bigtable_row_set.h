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

#ifndef BIGTABLE_ROW_SET_H
#define BIGTABLE_ROW_SET_H

#include "google/cloud/bigtable/table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_resource_kernel.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_row_range.h"

namespace tensorflow {
namespace io {

class BigtableRowSetResource : public ResourceBase {
 public:
  explicit BigtableRowSetResource(google::cloud::bigtable::RowSet row_set)
      : row_set_(std::move(row_set)) {
    VLOG(1) << "BigtableRowSetResource ctor";
  }

  ~BigtableRowSetResource() { VLOG(1) << "BigtableRowSetResource dtor"; }

  std::string ToString() const {
    std::string res;
    google::protobuf::TextFormat::PrintToString(row_set_.as_proto(), &res);
    return res;
  }

  void AppendRow(std::string const& row_key) { row_set_.Append(row_key); }
  void AppendRowRange(google::cloud::bigtable::RowRange const& row_range) {
    row_set_.Append(row_range);
  }
  google::cloud::bigtable::RowSet Intersect(
      google::cloud::bigtable::RowRange const& row_range) {
    return row_set_.Intersect(row_range);
  }

  google::cloud::bigtable::RowSet const& row_set() { return row_set_; }

  string DebugString() const override {
    return "BigtableRowSetResource:{" + ToString() + "}";
  }

 private:
  google::cloud::bigtable::RowSet row_set_;
};

}  // namespace io
}  // namespace tensorflow

#endif /* BIGTABLE_ROW_SET_H */
