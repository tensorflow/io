/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
namespace io {
namespace {

class IOGraphOptimizationPass : public GraphOptimizationPass {
 public:
  IOGraphOptimizationPass() {
    enable_ = (std::getenv("TFIO_GRAPH_DEBUG") != nullptr);
    if (enable_) {
      LOG(INFO) << "TFIO_GRAPH_DEBUG: [init]";
    }
  }
  virtual ~IOGraphOptimizationPass() {
    if (enable_) {
      LOG(INFO) << "TFIO_GRAPH_DEBUG: [fini]";
    }
  }
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (enable_) {
      Graph* graph = options.graph->get();
      LOG(INFO) << "TFIO_GRAPH_DEBUG: [run]:"
                << graph->ToGraphDefDebug().DebugString();
    }
    return Status::OK();
  }

 private:
  bool enable_ = false;
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 15,
                      IOGraphOptimizationPass);

}  // namespace
}  // namespace io
}  // namespace tensorflow
