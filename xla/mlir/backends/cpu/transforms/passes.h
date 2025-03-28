/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_CPU_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_CPU_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace xla {
namespace cpu {

//===----------------------------------------------------------------------===//
// Auxiliary passes for lowering to XLA Cpu runtime.
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLmhloToCpuRuntimePass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createXlaAbiLegalizationPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createLegalizeCollectiveOpsPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createLegalizeI1VectorTransferOpsPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createConvertXlaCpuMemRefElementCastToLLVMPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createRemoveCopiesToOutParamsPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createSparseCustomCallToPackUnpackOpPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createRewriteReallocToAllocPass();

//===-----------------------------------------------------------------------===/

#define GEN_PASS_REGISTRATION
#include "xla/mlir/backends/cpu/transforms/passes.h.inc"

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_CPU_TRANSFORMS_PASSES_H_
