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

#include "xla/mlir/tools/mlir_replay/mlir_replay_lib.h"

#include <complex>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/ParseUtilities.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"
#include "xla/mlir_hlo/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir_hlo/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace mlir {
namespace interpreter {
namespace {

tsl::StatusOr<SmallVector<InterpreterValue>> LoadArgs(
    const xla::HloSnapshot& snapshot) {
  SmallVector<InterpreterValue> result;
  for (const auto& arg : snapshot.arguments()) {
    TF_ASSIGN_OR_RETURN(auto converted, LiteralToValue(arg));
    result.push_back(std::move(converted));
  }
  return result;
}

namespace {
template <typename T, template <typename _> class rng_t>
mlir::interpreter::InterpreterValue RandomTensor(absl::BitGenRef bitgen,
                                                 mlir::Type type) {
  llvm::SmallVector<int64_t> shape;
  auto shaped_ty = type.dyn_cast<mlir::ShapedType>();
  if (shaped_ty) {
    shape = llvm::to_vector(shaped_ty.getShape());
  }

  auto rng = rng_t<T>{};
  auto result = mlir::interpreter::TensorOrMemref<T>::empty(shape);
  for (const auto& index : result.view.indices()) {
    auto& elem = result.at(index) = rng(bitgen);
    // Ints are typically indices, so scale them down to a more reasonable
    // range.
    if constexpr (std::is_same_v<T, int64_t>) {
      elem >>= 60;
    }
  }
  if (shaped_ty) {
    return {result};
  }
  return {result.at({})};
}
}  // namespace

mlir::FailureOr<mlir::interpreter::InterpreterValue> MakeRandomInput(
    absl::BitGenRef bitgen, mlir::Type type) {
  auto elem_ty =
      type.isa<ShapedType>() ? type.cast<ShapedType>().getElementType() : type;
  if (elem_ty.isF32()) {
    return RandomTensor<float, absl::gaussian_distribution>(bitgen, type);
  }
  if (elem_ty.isF64()) {
    return RandomTensor<double, absl::gaussian_distribution>(bitgen, type);
  }
  if (elem_ty.isInteger(32)) {
    return RandomTensor<int32_t, absl::uniform_int_distribution>(bitgen, type);
  }
  if (elem_ty.isInteger(16)) {
    return RandomTensor<int16_t, absl::uniform_int_distribution>(bitgen, type);
  }
  if (elem_ty.isInteger(64)) {
    return RandomTensor<int64_t, absl::uniform_int_distribution>(bitgen, type);
  }
  if (elem_ty.isInteger(1)) {
    return {{TensorOrMemref<bool>::empty(type.cast<ShapedType>().getShape())}};
  }

  llvm::errs() << "Unsupported type: ";
  type.print(llvm::errs());
  llvm::errs() << "\n";
  return failure();
}

}  // namespace

tsl::StatusOr<SmallVector<InterpreterValue>> Run(
    MLIRContext& context, const std::string& mlir_ir,
    const xla::HloSnapshot& snapshot, ExecutionTrace* trace,
    const std::string& entry) {
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlir_ir),
                                mlir::SMLoc());
  mlir::OwningOpRef<mlir::Operation*> module =
      mlir::parseSourceFileForTool(sourceMgr, &context, false);
  if (!module) {
    return tsl::errors::InvalidArgument("failed to parse MLIR");
  }

  SymbolTable symbols(*module);
  auto main = llvm::dyn_cast_or_null<func::FuncOp>(symbols.lookup(entry));
  if (!main) {
    return tsl::errors::InvalidArgument("failed to find entry function \"" +
                                        entry + "\"");
  }

  if (trace) {
    llvm::raw_string_ostream os(*trace->mutable_ir());
    (*module)->print(os, OpPrintingFlags().printGenericOpForm());
  }

  // After xla-rt-export-functions, we have an execution context as the first
  // argument. The interpreter currently cannot deal with these things, so we
  // fail in that case.
  auto function_args = main.getBody().getBlocks().front().getArguments();
  if (!llvm::all_of(function_args, [](Value arg) {
        return arg.getType().isa<ShapedType>();
      })) {
    return tsl::errors::InvalidArgument(
        "expected all function arguments to be shaped types");
  }

  TF_ASSIGN_OR_RETURN(auto args, LoadArgs(snapshot));
  auto out_args =
      main.getBody().getBlocks().front().getArguments().drop_front(args.size());

  std::seed_seq my_seed_seq({0});
  absl::BitGen bitgen(my_seed_seq);
  llvm::SmallVector<InterpreterValue> out_buffers;
  // Add random inputs for output arguments and unspecified inputs.
  for (auto arg : out_args) {
    auto arg_or = MakeRandomInput(bitgen, arg.getType());
    if (!succeeded(arg_or)) {
      return tsl::errors::InvalidArgument("failed to create input");
    }
    out_buffers.push_back(*arg_or);
    args.push_back(*arg_or);
  }

  InterpreterOptions options;
  ExecutionTraceListener tracer(trace);
  if (trace) {
    options.listener = &tracer;
  }
  auto results_or = runInterpreter(symbols, main, args, options);
  if (!succeeded(results_or)) {
    return tsl::errors::Internal("interpreter failed");
  }

  if (results_or->empty()) {
    return out_buffers;
  }
  return *results_or;
}

}  // namespace interpreter
}  // namespace mlir
