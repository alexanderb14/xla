/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/ir_emission_utils.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// Return whether the given shape is rank 2 excluding the batch dimensions.
bool IsRank2(const Shape& shape, int64_t batch_dimensions_size) {
  return shape.rank() == batch_dimensions_size + 2;
}

// Given a shape and a group of contiguous dimensions in the shape, returns
// a tuple of three values (major, middle, minor), where major is the size of
// the dimensions more major then the given dimensions, minor is the size of
// dimensions more minor then the given dimensions, and middle is the size of
// the given dimensions.
Vector3 PartitionShapeByMiddleDimensions(
    const Shape& shape, absl::Span<const int64_t> dims_middle) {
  CHECK(LayoutUtil::AreDimensionsConsecutive(shape.layout(), dims_middle));
  Vector3 values = {1, 1, 1};
  enum Segment { kMajor = 0, kMiddle = 1, kMinor = 2 };
  Segment cur_segment = kMinor;

  for (int64_t cur_dim : LayoutUtil::MinorToMajor(shape)) {
    if (cur_segment != kMajor) {
      // Handle change of segments.
      bool cur_dim_in_middle = absl::c_linear_search(dims_middle, cur_dim);
      if (cur_segment == kMinor) {
        if (cur_dim_in_middle) {
          cur_segment = kMiddle;
        }
      } else if (cur_segment == kMiddle) {
        if (!cur_dim_in_middle) {
          cur_segment = kMajor;
        }
      }
    }
    values[cur_segment] *= shape.dimensions(cur_dim);
  }
  return values;
}

Shape GetShapeFromTensorType(mlir::Value value) {
  constexpr char kDefaultLayoutAttrName[] = "xla_shape";

  mlir::Operation* op = value.getDefiningOp();
  CHECK(op);
  CHECK(value.getType().isa<mlir::TensorType>());
  Shape shape;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(kDefaultLayoutAttrName)) {
    shape = *xla::ParseShape(
        absl::string_view(attr.getValue().data(), attr.getValue().size()));
  } else {
    shape = TypeToShape(value.getType());
  }
  return shape;
}

}  // namespace

bool IsMatrixMultiplication(const HloInstruction& dot) {
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  PrimitiveType output_primitive_type = dot.shape().element_type();
  bool type_is_allowed =
      (output_primitive_type == F8E4M3FN || output_primitive_type == F8E5M2 ||
       output_primitive_type == F16 || output_primitive_type == BF16 ||
       output_primitive_type == F32 || output_primitive_type == F64 ||
       output_primitive_type == C64 || output_primitive_type == C128) ||
      (output_primitive_type == S32 && lhs_shape.element_type() == S8 &&
       lhs_shape.element_type() == S8);
  bool shapes_are_valid =
      type_is_allowed &&
      IsRank2(lhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
      IsRank2(rhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
      IsRank2(dot.shape(), dim_numbers.lhs_batch_dimensions_size()) &&
      !ShapeUtil::IsZeroElementArray(lhs_shape) &&
      !ShapeUtil::IsZeroElementArray(rhs_shape);

  if (!shapes_are_valid) {
    return false;
  }

  // The size of the reduction dimension should match. The shape inference
  // guarantees this invariant, so the check here is for programming
  // errors.
  CHECK_EQ(lhs_shape.dimensions(dim_numbers.lhs_contracting_dimensions(0)),
           rhs_shape.dimensions(dim_numbers.rhs_contracting_dimensions(0)));

  return true;
}

Vector3 GetReductionTiling(const ReductionDimensions& reduction_dimensions) {
  if (reduction_dimensions.is_row_reduction) {
    int64_t tile_z = std::min(reduction_dimensions.dimensions[0],
                              BatchedReductionRaceFreeBound());
    return {tile_z, 1, 16};
  }

  // Column reduction.
  return {1, 128, 1};
}

const char* const kCusolverCholeskyCallTarget = "__cusolver$cholesky";

bool IsCustomCallToCusolver(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  return hlo.custom_call_target() == kCusolverCholeskyCallTarget;
}

static bool IsUnnestedReductionFasterThanElemental(
    const ReductionDimensions& reduction_dimensions) {
  if (reduction_dimensions.is_row_reduction) {
    // For row reduction, the tile block is 1 x tile_size_x, and we are reducing
    // along tile_size_x which needs to be large enough to make the tiling
    // implementation efficient.
    // For very small reductions with a power-of-two size, we can fit multiple
    // reductions inside a single warp, which is more efficient than a loop.
    return (reduction_dimensions.dimensions[2] >= WarpSize()) ||
           ((WarpSize() % reduction_dimensions.dimensions[2]) == 0);
  }

  // For column reduction, the tile block is tile_size_y x tile_size_x, and we
  // are reducing along tile_size_y. Only tile_size_y needs to be
  // large enough to make the tiling implementation efficient.
  int64_t major_size = reduction_dimensions.dimensions[1];
  int64_t minor_size = reduction_dimensions.dimensions[2];

  // Rule generated by sweeping the search space of small column reductions.
  bool prefer_elemental_emitter =
      (major_size < WarpSize()) ||
      (major_size < 2 * WarpSize() && minor_size < WarpSize()) ||
      (major_size < 4 * WarpSize() && minor_size < 8) ||
      (major_size < 8 * WarpSize() && minor_size < 3);

  return !prefer_elemental_emitter;
}

bool IsReductionFromOrToContiguousDimensions(const HloInstruction& reduce) {
  if (reduce.opcode() != HloOpcode::kReduce) {
    return false;
  }

  const Shape& operand_shape = reduce.operand(0)->shape();
  absl::Span<const int64_t> dims_to_reduce = reduce.dimensions();
  DimensionVector dims_to_keep;
  for (int64_t dim = 0; dim < operand_shape.dimensions().size(); ++dim) {
    if (!absl::c_linear_search(dims_to_reduce, dim)) {
      dims_to_keep.push_back(dim);
    }
  }

  // We support fast codegen for three cases:
  // 1) Row reduction: (K, R)
  // 2) Column reduction: (K, R, K)
  // 3) "Batched" row reduction: (R, K, R)
  return (LayoutUtil::AreDimensionsConsecutive(operand_shape.layout(),
                                               dims_to_keep) ||
          LayoutUtil::AreDimensionsConsecutive(operand_shape.layout(),
                                               dims_to_reduce)) &&
         IsUnnestedReductionFasterThanElemental(
             GetReductionKindAndContiguousComponents(reduce));
}

bool IsInputFusibleSlices(mlir::Operation* unnested_hlo,
                          bool verify_no_strides) {
  auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(unnested_hlo);
  if (!fusion) {
    return false;
  }

  auto is_non_strided = [](mlir::DenseIntElementsAttr strides) -> bool {
    return absl::c_all_of(
        strides, [](const llvm::APInt& stride) { return stride == 1; });
  };

  for (mlir::Value value : fusion.getFusionResults()) {
    auto slice =
        mlir::dyn_cast_or_null<mlir::mhlo::SliceOp>(value.getDefiningOp());
    if (!slice) {
      return false;
    }
    if (verify_no_strides && !is_non_strided(slice.getStrides())) {
      return false;
    }
  }
  return true;
}

ReductionDimensions GetReductionKindAndContiguousComponents(
    const HloInstruction& reduce) {
  Shape input_shape = reduce.operand(0)->shape();
  absl::Span<const int64_t> dims_to_reduce = reduce.dimensions();
  DimensionVector dims_to_keep;
  for (int64_t dim = 0; dim < input_shape.rank(); ++dim) {
    if (!absl::c_linear_search(dims_to_reduce, dim)) {
      dims_to_keep.push_back(dim);
    }
  }

  if (dims_to_keep.empty()) {
    return {/*is_row_reduction=*/true,
            {1, 1, ShapeUtil::ElementsIn(input_shape)}};
  }

  if (LayoutUtil::AreDimensionsConsecutive(input_shape.layout(),
                                           dims_to_keep)) {
    Vector3 shape_partition =
        PartitionShapeByMiddleDimensions(input_shape, dims_to_keep);
    if (shape_partition[1] == 1) {
      return {/*is_row_reduction=*/true,
              {1, 1, shape_partition[0] * shape_partition[2]}};
    }
    if (shape_partition[2] == 1) {
      return {/*is_row_reduction=*/false,
              {1, shape_partition[0], shape_partition[1]}};
    }
    return {/*is_row_reduction=*/true, shape_partition};
  }

  Vector3 shape_partition =
      PartitionShapeByMiddleDimensions(input_shape, dims_to_reduce);

  if (shape_partition[2] == 1) {
    return {/*is_row_reduction=*/true,
            {1, shape_partition[0], shape_partition[1]}};
  }
  return {/*is_row_reduction=*/false, shape_partition};
}

// This emits a device-side call to
// "i32 vprintf(i8* fmt, arguments_type* arguments)" in the driver; see
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
llvm::Value* EmitPrintf(absl::string_view fmt,
                        absl::Span<llvm::Value* const> arguments,
                        llvm::IRBuilder<>* builder) {
  std::vector<llvm::Type*> argument_types;

  // Variadic arguments implicit promotion [1] converts float to double,
  // and bool/char/short are converted to int.
  // [1] https://en.cppreference.com/w/cpp/language/variadic_arguments
  auto requires_int32_promotion = [](llvm::Type* type) {
    return type->isIntegerTy(/*BitWidth=*/1) ||
           type->isIntegerTy(/*BitWidth=*/8) ||
           type->isIntegerTy(/*BitWidth=*/16);
  };
  auto requires_double_promotion = [](llvm::Type* type) {
    return type->isFloatingPointTy();
  };

  for (auto argument : arguments) {
    llvm::Type* type = argument->getType();
    if (requires_double_promotion(type)) {
      argument_types.push_back(builder->getDoubleTy());
    } else if (requires_int32_promotion(type)) {
      argument_types.push_back(builder->getInt32Ty());
    } else {
      argument_types.push_back(type);
    }
  }
  auto* arguments_type = llvm::StructType::create(argument_types);
  llvm::Value* arguments_ptr = builder->CreateAlloca(arguments_type);
  for (size_t i = 0; i < arguments.size(); ++i) {
    llvm::Value* value = arguments[i];
    llvm::Type* type = value->getType();
    if (requires_double_promotion(type)) {
      value = builder->CreateFPCast(value, builder->getDoubleTy());
    } else if (requires_int32_promotion(type)) {
      value = builder->CreateIntCast(value, builder->getInt32Ty(),
                                     /*isSigned=*/true);
    }
    builder->CreateStore(
        value,
        builder->CreateGEP(arguments_type, arguments_ptr,
                           {builder->getInt64(0), builder->getInt32(i)}));
  }
  llvm::Type* ptr_ty = builder->getInt8Ty()->getPointerTo();
  return builder->CreateCall(
      builder->GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
          "vprintf",
          llvm::FunctionType::get(builder->getInt32Ty(), {ptr_ty, ptr_ty},
                                  /*isVarArg=*/false)),
      {builder->CreateGlobalStringPtr(llvm_ir::AsStringRef(fmt)),
       builder->CreatePointerCast(arguments_ptr, ptr_ty)});
}

// Helper function to emit call to AMDGPU shfl_down function.
llvm::Value* EmitAMDGPUShflDown(llvm::Value* value, llvm::Value* offset,
                                llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  CHECK_EQ(value->getType()->getPrimitiveSizeInBits(), 32);
  auto* i32_ty = b->getInt32Ty();
  llvm::FunctionCallee shfl_fn = module->getOrInsertFunction(
      llvm_ir::AsStringRef("__ockl_readuplane_i32"),
      llvm::FunctionType::get(/*Result=*/i32_ty, {i32_ty, i32_ty},
                              /*isVarArg=*/false));
  // AMDGPU device function requires first argument as i32.
  llvm::Value* result =
      b->CreateCall(shfl_fn, {b->CreateBitCast(value, i32_ty), offset});
  // AMDGPU device function always returns an i32 type.
  return b->CreateBitCast(result, value->getType());
}

// Helper function to emit call to NVPTX shfl_down intrinsic.
llvm::Value* EmitNVPTXShflDown(llvm::Value* value, llvm::Value* offset,
                               llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Intrinsic::ID llvm_intrinsic_id;
  CHECK_EQ(value->getType()->getPrimitiveSizeInBits(), 32);
  if (value->getType()->isFloatTy()) {
    llvm_intrinsic_id = llvm::Intrinsic::nvvm_shfl_sync_down_f32;
  } else {
    llvm_intrinsic_id = llvm::Intrinsic::nvvm_shfl_sync_down_i32;
  }
  llvm::Function* intrinsic =
      llvm::Intrinsic::getDeclaration(module, llvm_intrinsic_id, {});
  return b->CreateCall(
      intrinsic, {b->getInt32(-1), value, offset, b->getInt32(WarpSize() - 1)});
}

llvm::Value* EmitFullWarpShuffleDown(llvm::Value* value, llvm::Value* offset,
                                     llvm::IRBuilder<>* builder) {
  int bit_width = value->getType()->getPrimitiveSizeInBits();
  llvm::Module* module = builder->GetInsertBlock()->getModule();
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());

  // Special case for efficiency
  if (value->getType()->isFloatTy() && bit_width == 32) {
    if (target_triple.isNVPTX()) {
      return EmitNVPTXShflDown(value, offset, builder);
    } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
      return EmitAMDGPUShflDown(value, offset, builder);
    } else {
      LOG(FATAL) << "Invalid triple " << target_triple.str();
    }
  }

  // We must split values wider than 32 bits as the "shfl" instruction operates
  // on 32-bit values.
  int num_segments = CeilOfRatio(bit_width, 32);
  llvm::Value* x = builder->CreateBitCast(
      builder->CreateZExt(
          builder->CreateBitCast(value, builder->getIntNTy(bit_width)),
          builder->getIntNTy(32 * num_segments)),
      llvm::VectorType::get(builder->getInt32Ty(), num_segments, false));
  for (int i = 0; i < num_segments; ++i) {
    llvm::Value* insert_val;
    if (target_triple.isNVPTX()) {
      insert_val = EmitNVPTXShflDown(builder->CreateExtractElement(x, i),
                                     offset, builder);
    } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
      insert_val = EmitAMDGPUShflDown(builder->CreateExtractElement(x, i),
                                      offset, builder);
    } else {
      LOG(FATAL) << "Invalid triple " << target_triple.str();
    }
    x = builder->CreateInsertElement(x, insert_val, i);
  }
  return builder->CreateBitCast(
      builder->CreateTrunc(
          builder->CreateBitCast(x, builder->getIntNTy(32 * num_segments)),
          builder->getIntNTy(bit_width)),
      value->getType());
}

llvm::Value* IsBlock0Thread0(llvm::IRBuilder<>* b) {
  llvm::Value* is_thread0 = b->CreateICmpEQ(
      b->getInt32(0),
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b));

  llvm::Value* is_block0 = b->CreateICmpEQ(
      b->getInt32(0),
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b));
  return b->CreateAnd(is_thread0, is_block0);
}

// Given an LMHLO op, returns the operand index of the first output operand.
//
// Notice that an operand alised to an output isn't an output, even though in
// that case WritesMlirBuffer() returns true on that operand.
//
// An operand is !WritesMlirBuffer() || equals (aliases) to a later operand. An
// output is the opposite, being both WritesMlirBuffer() and does not equal to
// any later operand.
int PartitionLmhloOperandsAndOutputs(mlir::Operation* op) {
  CHECK(op->getDialect() == op->getContext()->getLoadedDialect("lmhlo"));

  int i;
  for (i = op->getOperands().size() - 1; i >= 0; i--) {
    const bool aliased =
        std::find(op->getOperands().begin() + i + 1, op->getOperands().end(),
                  op->getOperand(i)) != op->getOperands().end();
    if (!WritesMlirBuffer(op, op->getOperand(i)) || aliased) {
      break;
    }
  }
  return i + 1;
}

llvm::SmallVector<mlir::Value> GetHloOperands(mlir::Operation* op) {
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    return fusion.getInputBuffers();
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("lmhlo")) {
    int output_start = PartitionLmhloOperandsAndOutputs(op);
    llvm::SmallVector<mlir::Value> operands;
    for (int i = 0; i < output_start; i++) {
      operands.push_back(op->getOperand(i));
    }
    return operands;
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("mhlo")) {
    return op->getOperands();
  }
  LOG(FATAL) << "Unexpected op: " << llvm_ir::DumpToString(op);
}

llvm::SmallVector<mlir::Value> GetHloOutputs(mlir::Operation* op) {
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    return fusion.getOutputBuffers();
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("lmhlo")) {
    int output_start = PartitionLmhloOperandsAndOutputs(op);
    llvm::SmallVector<mlir::Value> outputs;
    for (int i = output_start; i < op->getNumOperands(); i++) {
      outputs.push_back(op->getOperand(i));
    }
    return outputs;
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("mhlo")) {
    return op->getResults();
  }
  LOG(FATAL) << "Unexpected op: " << llvm_ir::DumpToString(op);
}

bool WritesMlirBuffer(mlir::Operation* op, mlir::Value operand) {
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> effects;
  mlir::cast<mlir::MemoryEffectOpInterface>(op).getEffectsOnValue(operand,
                                                                  effects);
  return absl::c_any_of(
      effects, [](const mlir::MemoryEffects::EffectInstance& instance) {
        return mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
      });
}

static int64_t GetMemRefSizeInBytes(mlir::MemRefType type) {
  // For i1 memrefs, the underlying allocation is 8 bits.
  if (type.getElementType().isInteger(/*width=*/1)) {
    return type.getNumElements();
  } else {
    return type.cast<mlir::ShapedType>().getSizeInBits() / CHAR_BIT;
  }
}

static int64_t GetAllocationIndex(mlir::BlockArgument func_arg,
                                  std::string* constant_name) {
  auto func_op =
      mlir::cast<mlir::func::FuncOp>(func_arg.getParentRegion()->getParentOp());
  if (constant_name) {
    if (auto constant_name_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
            func_arg.getArgNumber(), "lmhlo.constant_name")) {
      *constant_name = constant_name_attr.getValue().str();
    }
  }
  return func_arg.getArgNumber();
}

StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    mlir::Value v, absl::Span<const BufferAllocation> allocations,
    std::string* constant_name) {
  if (constant_name) {
    constant_name->clear();
  }

  int64_t size = GetMemRefSizeInBytes(v.getType().cast<mlir::MemRefType>());

  // We match the following patterns here:
  //  base := ViewOp(arg) | get_global_memref (global_memref) | arg
  //  root := base | MemRefReinterpretCastOp(base) | CollapseShapeOp(base)

  if (auto cast = mlir::dyn_cast_or_null<mlir::memref::ReinterpretCastOp>(
          v.getDefiningOp())) {
    v = cast.getViewSource();
  }
  if (auto collapse_shape =
          mlir::dyn_cast_or_null<mlir::memref::CollapseShapeOp>(
              v.getDefiningOp())) {
    v = collapse_shape.getSrc();
  }

  if (auto view =
          mlir::dyn_cast_or_null<mlir::memref::ViewOp>(v.getDefiningOp())) {
    TF_RET_CHECK(view.getSource().isa<mlir::BlockArgument>());

    return BufferAllocation::Slice(
        &allocations[GetAllocationIndex(
            view.getSource().cast<mlir::BlockArgument>(), constant_name)],
        mlir::cast<mlir::arith::ConstantOp>(view.getByteShift().getDefiningOp())
            .getValue()
            .cast<mlir::IntegerAttr>()
            .getValue()
            .getSExtValue(),
        size);
  }
  if (auto get_global = mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(
          v.getDefiningOp())) {
    auto module = get_global->getParentOfType<mlir::ModuleOp>();
    if (constant_name) {
      *constant_name = get_global.getName().str();
    }
    auto global = mlir::cast<mlir::memref::GlobalOp>(
        module.lookupSymbol(get_global.getName()));
    int64_t index =
        global->getAttrOfType<mlir::IntegerAttr>("lmhlo.alloc").getInt();
    return BufferAllocation::Slice(&allocations[index], 0,
                                   allocations[index].size());
  }
  if (auto arg = v.dyn_cast<mlir::BlockArgument>()) {
    return BufferAllocation::Slice(
        &allocations[GetAllocationIndex(arg, constant_name)], 0, size);
  }

  return Unimplemented(
      "Operand has to be in the form of ViewOp(arg) or "
      "StaticMemRefCastOp(ViewOp(arg)) or arg");
}

bool CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
    mlir::lmhlo::FusionOp fusion,
    absl::Span<const BufferAllocation> allocations) {
  auto results = fusion.getFusionResults();
  if (results.size() != 1) {
    return false;
  }
  auto dus = mlir::dyn_cast<mlir::mhlo::DynamicUpdateSliceOp>(
      results[0].getDefiningOp());
  if (!dus) {
    return false;
  }

  auto output_buffers = fusion.getOutputBuffers();
  CHECK_EQ(1, output_buffers.size());
  auto parameter = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(
      dus.getOperand().getDefiningOp());

  if (!parameter) {
    return false;
  }

  auto maybe_lhs = GetAllocationSlice(parameter.getMemref(), allocations);
  auto maybe_rhs = GetAllocationSlice(output_buffers[0], allocations);
  return maybe_lhs.ok() && maybe_rhs.ok() && *maybe_lhs == *maybe_rhs;
}

Shape GetShape(mlir::Value value) {
  if (value.getType().isa<mlir::MemRefType>()) {
    return TypeToShape(value.getType());
  } else if (value.getType().isa<mlir::TensorType>()) {
    return GetShapeFromTensorType(value);
  } else if (value.getType().isa<mlir::TupleType>()) {
    return TypeToShape(value.getType());
  }
  LOG(FATAL) << "Unexpected value type to get shape for";
  return {};
}

bool ReductionIsRaceFree(const ReductionDimensions& reduction_dimensions) {
  Vector3 reduction_tiling = GetReductionTiling(reduction_dimensions);
  return (reduction_dimensions.is_row_reduction &&
          reduction_dimensions.dimensions[2] <=
              MinThreadsXRowReduction() * reduction_tiling[2] &&
          reduction_dimensions.dimensions[0] <=
              BatchedReductionRaceFreeBound()) ||
         (!reduction_dimensions.is_row_reduction &&
          reduction_dimensions.dimensions[1] <=
              WarpSize() * reduction_tiling[1]);
}

// Recursive helper for GetFusionRoots below.
static void GetFusionRootsRec(HloInstruction* root,
                              std::vector<HloInstruction*>& out) {
  if (root->opcode() == HloOpcode::kGetTupleElement) {
    return GetFusionRootsRec(root->mutable_operand(0), out);
  } else if (root->opcode() == HloOpcode::kTuple) {
    for (int i = 0; i < root->operand_count(); i++) {
      GetFusionRootsRec(root->mutable_operand(i), out);
    }
  } else {
    if (!out.empty() && out.back() == root) {
      return;
    }
    CHECK(!absl::c_linear_search(out, root))
        << "Fusion root contains instruction " << root->ToString()
        << " multiple times";
    out.push_back(root);
  }
}

std::vector<HloInstruction*> GetFusionRoots(HloComputation* computation) {
  std::vector<HloInstruction*> out;
  GetFusionRootsRec(computation->root_instruction(), out);
  return out;
}

static std::optional<Vector3> FindTiledTranspose(const HloInstruction& instr,
                                                 Vector3& permutation) {
  if (instr.opcode() != HloOpcode::kCopy) {
    return std::nullopt;
  }

  if (std::optional<Vector3> tr = ShapeUtil::GetNormalizedTransposeShape(
          instr.operand(0)->shape(), instr.shape(), Vector3{0, 2, 1})) {
    if (tr->at(1) >= kMinDimensionToTransposeTiled &&
        tr->at(2) >= kMinDimensionToTransposeTiled) {
      permutation = Vector3{0, 2, 1};
      return tr;
    }
  }
  if (std::optional<Vector3> tr = ShapeUtil::GetNormalizedTransposeShape(
          instr.operand(0)->shape(), instr.shape(), Vector3{2, 1, 0})) {
    if (tr->at(0) >= kMinDimensionToTransposeTiled &&
        tr->at(2) >= kMinDimensionToTransposeTiled) {
      permutation = Vector3{2, 1, 0};
      return tr;
    }
  }
  return std::nullopt;
}

// Find 021 or 210 transpose in logical + physical transposition.
std::optional<Vector3> FindTiledLogicalTranspose(const HloInstruction& instr,
                                                 Vector3& permutation) {
  if (instr.opcode() != HloOpcode::kTranspose) {
    return std::nullopt;
  }

  // TODO(cheshire): avoid code duplication.
  if (std::optional<Vector3> tr = ShapeUtil::GetNormalizedLogicalTransposeShape(
          instr.operand(0)->shape(), instr.shape(), instr.dimensions(),
          Vector3{0, 2, 1})) {
    if (tr->at(1) >= kMinDimensionToTransposeTiled &&
        tr->at(2) >= kMinDimensionToTransposeTiled) {
      permutation = Vector3{0, 2, 1};
      return tr;
    }
  }
  if (std::optional<Vector3> tr = ShapeUtil::GetNormalizedLogicalTransposeShape(
          instr.operand(0)->shape(), instr.shape(), instr.dimensions(),
          Vector3{2, 1, 0})) {
    if (tr->at(0) >= kMinDimensionToTransposeTiled &&
        tr->at(2) >= kMinDimensionToTransposeTiled) {
      permutation = Vector3{2, 1, 0};
      return tr;
    }
  }
  return std::nullopt;
}

std::optional<std::pair<Vector3, Vector3>> FindAnyTiledTranspose(
    const HloInstruction& instr) {
  const HloInstruction& hero = FindNonTrivialHero(instr);

  Vector3 permutation;
  if (std::optional<Vector3> d1 = FindTiledTranspose(hero, permutation)) {
    return std::make_pair(d1.value(), permutation);
  }
  if (std::optional<Vector3> d2 =
          FindTiledLogicalTranspose(hero, permutation)) {
    return std::make_pair(d2.value(), permutation);
  }
  return std::nullopt;
}

static bool IsIntermediate(const HloInstruction* instr) {
  return (
      instr->operand_count() == 1 && instr->user_count() <= 1 &&
      ((instr->IsElementwise() && instr->opcode() != HloOpcode::kCopy) ||
       instr->opcode() == HloOpcode::kBitcast ||
       (instr->opcode() == HloOpcode::kReshape &&
        ShapeUtil::ReshapeIsBitcast(instr->operand(0)->shape(),
                                    instr->shape())) ||
       (instr->opcode() == HloOpcode::kTranspose &&
        ShapeUtil::TransposeIsBitcast(instr->operand(0)->shape(),
                                      instr->shape(), instr->dimensions()))));
}

const HloInstruction& FindNonTrivialHero(const HloInstruction& instr) {
  const HloInstruction* idx = &instr;

  // Go up the chain of trivial elementwise(+bitcast, -copy) operations. Such
  // chains are bound to be quite small, as we restrict the number of users as
  // well. Note that no memoization is needed due to user number constraints: we
  // never have to revisit same nodes.
  while (IsIntermediate(idx)) {
    idx = idx->operand(0);
  }
  return *idx;
}

bool HasAnyTiledTransposeRoot(HloComputation* computation) {
  return absl::c_any_of(GetFusionRoots(computation),
                        [&](const HloInstruction* instr) {
                          return FindAnyTiledTranspose(*instr);
                        });
}

bool HasAnyUnnestedReductionRoot(HloComputation* computation) {
  return absl::c_any_of(
      GetFusionRoots(computation), [&](const HloInstruction* instr) {
        return IsReductionFromOrToContiguousDimensions(*instr);
      });
}

void LogAndVerify(const llvm::Module* m) {
  if (VLOG_IS_ON(5)) {
    XLA_VLOG_LINES(5, llvm_ir::DumpToString(m));
  }

  std::string llir_str;
  llvm::raw_string_ostream llir_stream(llir_str);
  bool broken = llvm::verifyModule(*m, &llir_stream);
  llir_stream.flush();
  CHECK(!broken) << llir_str;
}

}  // namespace gpu
}  // namespace xla
