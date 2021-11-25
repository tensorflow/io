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

#ifndef BIGTABLE_VERSION_FILTERS_H
#define BIGTABLE_VERSION_FILTERS_H

#include "absl/memory/memory.h"
#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_resource_kernel.h"

namespace tensorflow {
namespace io {

class BigtableFilterResource : public ResourceBase {
 public:
  explicit BigtableFilterResource(google::cloud::bigtable::Filter filter)
      : filter_(std::move(filter)) {
    VLOG(1) << "BigtableFilterResource ctor";
  }

  ~BigtableFilterResource() { VLOG(1) << "BigtableFilterResource dtor"; }

  std::string ToString() const {
    std::string res;
    google::protobuf::TextFormat::PrintToString(filter_.as_proto(), &res);
    return res;
  }

  const google::cloud::bigtable::Filter& filter() const { return filter_; }

  string DebugString() const override {
    return "BigtableFilterResource:{" + ToString() + "}";
  }

 private:
  const google::cloud::bigtable::Filter filter_;
};

}  // namespace io
}  // namespace tensorflow

#endif /* BIGTABLE_ROW_SET_H */
