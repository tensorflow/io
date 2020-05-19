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

#include "mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/IR/Module.h"              // from @llvm-project
#include "mlir/Pass/Pass.h"              // from @llvm-project
#include "mlir/Pass/PassManager.h"       // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace tensorflow {
namespace io {
namespace {

class MlirIOGraphOptimizationPass : public ::tensorflow::MlirOptimizationPass {
 public:
  llvm::StringRef name() const override { return "io_graph_optimization"; }

  bool IsEnabled(const ::tensorflow::ConfigProto& config_proto) const override {
    if (std::getenv("TFIO_GRAPH_DEBUG") == nullptr) {
      VLOG(1) << "Skipping MLIR IO Graph Optimization Pass"
              << ", TFIO_GRAPH_DEBUG not enabled";
      return false;
    }
    return true;
  }

  ::tensorflow::Status Run(const ::tensorflow::ConfigProto& config_proto,
                           mlir::ModuleOp module) override {
    if (std::getenv("TFIO_GRAPH_DEBUG") == nullptr) {
      VLOG(1) << "Skipping MLIR IO Graph Optimization Pass"
              << ", TFIO_GRAPH_DEBUG not enabled";
      return Status::OK();
    }

    VLOG(1) << "Run IO MLIR Graph Optimization Passes";
    mlir::PassManager pm(module.getContext());
    std::string str;
    llvm::raw_string_ostream os(str);
    module.print(os);
    LOG(INFO) << "IO Graph: " << os.str();

    // pm.addPass(createTODOPass());
    // mlir::LogicalResult result = pm.run(module);

    return Status::OK();
  }
};

static mlir_pass_registration::MlirOptimizationPassRegistration
    register_mlir_graph_optimization_pass(
        10, std::make_unique<MlirIOGraphOptimizationPass>());

}  // namespace
}  // namespace io
}  // namespace tensorflow
