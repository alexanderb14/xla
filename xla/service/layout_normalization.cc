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

#include "xla/service/layout_normalization.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/permutation_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Layout normalization visitor. Aims to achieve the global postcondition that
// every layout is strictly descending (the layout permutation is effectively
// applied to the shape itself).
//
// Local precondition for every call:
//    -> Input is a bitcast from a normalized layout.
//
// Local postcondition:
//    -> Input and output of a processed operation have descending layout*
//
// *: For current fusion limitations this is currently not applicable to
// unnested reductions only.
class LayoutNormalizationVisitor : public DfsHloRewriteVisitor {
 public:
  explicit LayoutNormalizationVisitor(
      const CustomCallTransformer& custom_call_transformer = nullptr)
      : custom_call_transformer_(custom_call_transformer) {}

  // To handle a constant, just give the literal data a new layout.
  Status HandleConstant(HloInstruction* hlo) override {
    const Literal& literal = hlo->literal();
    const Shape& shape = hlo->shape();
    if (literal.shape().IsTuple()) {
      // TODO(cheshire): Tuple constants.
      return OkStatus();
    }

    Shape normalized_shape = Normalize(hlo->shape());

    Literal new_literal(normalized_shape);

    // TODO(cheshire): Do not duplicate storage.
    std::memcpy(new_literal.untyped_data(), literal.untyped_data(),
                literal.size_bytes());

    HloInstruction* normalized = hlo->parent()->AddInstruction(
        HloInstruction::CreateConstant(std::move(new_literal)),
        &hlo->metadata());
    HloInstruction* bc_to_orig = MakeBitcastHlo(normalized, shape);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // Slice is layout-preserving, so handling is analoguous to elementwise unary,
  // and transposing the elements inside the metadata.
  Status HandleSlice(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    const Shape& s = hlo->shape();
    const Shape& operand_shape = operand->shape();
    TF_RET_CHECK(s.layout() == operand_shape.layout());
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));

    Shape normalized_w_degen =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            operand_shape);

    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());

    auto normalize_slice_attr = [&](absl::Span<int64_t const> input) {
      return PermuteSliceAttributes(input, layout_as_permutation,
                                    normalized_w_degen);
    };

    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_slice,
                        MakeSliceHlo(normalized_input,
                                     normalize_slice_attr(hlo->slice_starts()),
                                     normalize_slice_attr(hlo->slice_limits()),
                                     normalize_slice_attr(hlo->slice_strides()),
                                     &hlo->metadata()));
    *normalized_slice->mutable_shape()->mutable_layout() =
        normalized_input->shape().layout();
    Shape normalized_shape = Normalize(s);

    // Output of slice might contain degenerate dimensions.
    HloInstruction* bc_to_normalized =
        MakeBitcastHlo(normalized_slice, normalized_shape);
    HloInstruction* bc_to_orig = MakeBitcastHlo(bc_to_normalized, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // Default action: ensure local postcondition that any input is always a
  // bitcast from canonical layout for any rewrites of the HLO users.
  //
  // Bitcast to descending layout and then bitcast back to make sure that shapes
  // match.
  Status DefaultAction(HloInstruction* hlo) override {
    if (!hlo->user_count()) {
      // The local postcondition does not have to apply to the case when there
      // are no users.
      return OkStatus();
    }
    auto users = hlo->users();
    auto shape = hlo->shape();
    if (shape.IsTuple() || shape.IsToken()) {
      // GTEs will be transformed individually, tokens should be skipped.
      return OkStatus();
    }

    auto normalized_shape = Normalize(shape);
    auto bc_to_normalized = MakeBitcastHlo(hlo, normalized_shape);
    auto bc_to_orig = MakeBitcastHlo(bc_to_normalized, shape);
    TF_RETURN_IF_ERROR(hlo->ReplaceUsesWith(users, bc_to_orig));
    MarkAsChanged();
    return OkStatus();
  }

  // Converts concatenation to normalized layout.
  //
  // With respect to layouts, concatenations are simple, as they are
  // layout-preserving. However, there are some complications with respect to
  // degenerate dimensions: since our normalized form drops degenerate
  // dimensions, that might make the concatenation impossible, as the
  // corresponding concatenated dimension might not exist in the normalized
  // form.
  //
  // So we drop all degenerate dimensions EXCEPT for the one being concatenated.
  Status HandleConcatenate(HloInstruction* hlo) override {
    const Shape& s = hlo->shape();
    int64_t orig_concat_dim = hlo->dimensions(0);

    std::vector<HloInstruction*> normalized_inputs;
    for (HloInstruction* operand : hlo->mutable_operands()) {
      TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));
      const Shape& normalized_input_s = normalized_input->shape();
      const Shape& operand_s = operand->shape();

      // Drop all degenerate dimensions, unless it is being concatenated.
      auto operand_s_filtered = ShapeUtil::FilterDimensions(
          [&](int dim) {
            return operand_s.dimensions(dim) != 1 || dim == orig_concat_dim;
          },
          operand_s);

      auto operand_s_normalized =
          ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
              operand_s_filtered);
      auto new_operand =
          operand_s_normalized == normalized_input_s
              ? normalized_input
              : MakeBitcastHlo(normalized_input, operand_s_normalized);
      normalized_inputs.push_back(new_operand);
    }

    auto out_shape_degen_dropped = ShapeUtil::FilterDimensions(
        [&](int dim) {
          return s.dimensions(dim) != 1 || dim == orig_concat_dim;
        },
        s);
    auto normalized_w_degen =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(s);
    auto normalized_shape =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            out_shape_degen_dropped);

    auto l = ToTransposeDimensions(s.layout());
    int64_t normalized_concat_dim = FindIndex(l, orig_concat_dim);
    auto degen_delta = absl::c_count_if(
        normalized_w_degen.dimensions().subspan(0, normalized_concat_dim),
        [&](int dim) { return dim == 1; });
    auto normalized_concat = hlo->AddInstruction(
        HloInstruction::CreateConcatenate(normalized_shape, normalized_inputs,
                                          normalized_concat_dim - degen_delta));
    auto bc_to_orig = MakeBitcastHlo(normalized_concat, hlo->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  Status HandleReduceWindow(HloInstruction* hlo) override {
    if (hlo->shape().IsTuple()) {
      // TODO(cheshire): Handle variadic reductions.
      return OkStatus();
    }

    HloInstruction* operand = hlo->mutable_operand(0);
    TF_RET_CHECK(hlo->shape().layout() == operand->shape().layout());
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));

    HloInstruction* new_op;

    // TODO(cheshire): Try to have less duplication.
    const Shape& op_shape = operand->shape();
    Shape op_shape_reordered =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(op_shape);
    if (op_shape_reordered == normalized_input->shape()) {
      new_op = normalized_input;
    } else {
      new_op = MakeBitcastHlo(normalized_input, op_shape_reordered);
    }

    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());

    std::vector<WindowDimension> window_dimensions;
    for (const WindowDimension& d : hlo->window().dimensions()) {
      window_dimensions.push_back(d);
    }
    window_dimensions = Permute(window_dimensions, layout_as_permutation);

    Window new_window;
    for (const WindowDimension& d : window_dimensions) {
      *new_window.add_dimensions() = d;
    }

    TF_ASSIGN_OR_RETURN(
        HloInstruction * rw,
        MakeReduceWindowHlo(new_op, hlo->mutable_operand(1), new_window,
                            hlo->called_computations()[0], &hlo->metadata()));

    HloInstruction* bc_to_normalized =
        MakeBitcastHlo(rw, Normalize(rw->shape()));
    HloInstruction* bc_to_orig = MakeBitcastHlo(bc_to_normalized, hlo->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // Converts broadcast input and output to normalized layout.
  //
  // Converts:
  //
  //  A{I} -> bitcast{L} -> broadcast[S]{L'}
  //
  // Into:
  //
  //  A{I} -> broadcast[S']{I} -> bitcast[S]{L'}
  Status HandleBroadcast(HloInstruction* hlo) override {
    VLOG(3) << "Input broadcast: " << hlo->ToString();
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));
    auto normalized_shape = Normalize(s);
    std::vector<int64_t> orig_br_dimensions =
        NoDegenerateDims(hlo->dimensions(), operand->shape(), s);
    std::vector<int64_t> layout_as_permutation = ToTransposeDimensions(
        ShapeUtil::DropDegenerateDimensions(operand->shape()).layout());
    std::vector<int64_t> orig_output_layout_as_permutation =
        ToTransposeDimensions(ShapeUtil::DropDegenerateDimensions(s).layout());
    std::vector<int64_t> br_dimensions;
    if (!hlo->dimensions().empty()) {
      br_dimensions = Permute(orig_br_dimensions, layout_as_permutation);
    }
    for (int64_t& d : br_dimensions) {
      d = FindIndex(orig_output_layout_as_permutation, d);
    }
    absl::c_sort(br_dimensions);
    auto normalized_broadcast = MakeBroadcastHlo(
        normalized_input, br_dimensions, normalized_shape, &hlo->metadata());
    VLOG(3) << "Generated broadcast: " << normalized_broadcast->ToString();
    auto bc_to_orig = MakeBitcastHlo(normalized_broadcast, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // Pushes down the bitcast across the unary.
  // That is, converts:
  //
  //    H_0{I} -> B{L} -> U{L}
  //
  // into
  //
  //    H_0{I} -> U{I} -> B{L}
  //
  // where {I} denotes default layout.
  Status HandleElementwiseUnary(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    auto operand_shape = operand->shape();

    // Precondition: elementwise unary leaves layout intact.
    TF_RET_CHECK(s.layout() == operand_shape.layout())
        << "Unexpected non-layout preserving elementwise unary: "
        << hlo->ToString();
    TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));

    PrimitiveType to_element_type = s.element_type();
    HloInstruction* new_unary;
    if (hlo->opcode() == HloOpcode::kConvert) {
      new_unary =
          MakeConvertToHlo(normalized_input, to_element_type, &hlo->metadata());
    } else if (hlo->opcode() == HloOpcode::kReducePrecision) {
      new_unary =
          MakeReducePrecisionHlo(normalized_input, hlo->exponent_bits(),
                                 hlo->mantissa_bits(), &hlo->metadata());
    } else if (hlo->opcode() == HloOpcode::kBitcastConvert) {
      new_unary = MakeBitcastConvertToHlo(normalized_input, to_element_type,
                                          &hlo->metadata());
    } else {
      TF_ASSIGN_OR_RETURN(
          new_unary,
          MakeUnaryHlo(hlo->opcode(), normalized_input, &hlo->metadata()));
    }
    auto bc_to_orig = MakeBitcastHlo(new_unary, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // Pushes down the bitcast across the binary. Converts:
  //
  //  A1{I} -> bitcast{L}
  //            \
  //            B{L}
  //            /
  //  A2{I} -> bitcast{L}
  //
  // Into:
  //
  //  A1{I}
  //        \
  //         B{I} - bitcast{L}
  //        /
  //  A2{I}
  Status HandleElementwiseBinary(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto a = hlo->mutable_operand(0);
    auto b = hlo->mutable_operand(1);
    TF_RET_CHECK(a->shape().layout() == s.layout());
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(a));
    TF_ASSIGN_OR_RETURN(auto b0, GetNormalizedInput(b));

    HloInstruction* new_binary;
    if (hlo->opcode() == HloOpcode::kCompare) {
      TF_ASSIGN_OR_RETURN(new_binary,
                          MakeCompareHlo(hlo->comparison_direction(), a0, b0,
                                         &hlo->metadata()));
    } else {
      TF_ASSIGN_OR_RETURN(
          new_binary, MakeBinaryHlo(hlo->opcode(), a0, b0, &hlo->metadata()));
    }
    auto bc_to_orig = MakeBitcastHlo(new_binary, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // The ReshapeDecomposer already gives us a precondition that a reshape is
  // bitcast. Converts:
  //
  // A{I} -> bitcast [S0]{L1} -> R [S]{L2}
  //
  // Into:
  //
  // A{I} -> R [S']{I} -> bitcast[S]{L2}
  //
  Status HandleReshape(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_RET_CHECK(ShapeUtil::ReshapeIsBitcast(s, operand->shape()));
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto normalized_reshape_s = Normalize(s);
    TF_ASSIGN_OR_RETURN(auto new_reshape,
                        MakeReshapeHlo(normalized_reshape_s, a0));
    auto bc_to_orig = MakeBitcastHlo(new_reshape, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // For bitcasting transposes, converts:
  //
  // A{I} -> bitcast[S]{L} -> transpose{L2}
  //
  // Into:
  //
  // A{I} -> bitcast{L2}
  //
  // For non-bitcasting ones, converts:
  //
  // A{I} -> bitcast[S0]{L} -> transpose[S]{L2}
  //
  // Into:
  //
  // A{I} -> transpose[S']{I} -> bitcast{L2}
  //
  // Where S' is the normalization of [S]{L2}, and `dimensions` attribute is
  //
  // The `dimensions` of the new transposition is given by:
  //
  //  L^-1 o `dim_0` o L2
  //
  // where dim_0 is dimensions of the original transposition, and `o` denotes
  // permutation composition.
  Status HandleTranspose(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    auto operand_s = operand->shape();
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto normalized_shape = Normalize(s);
    VLOG(3) << "Input transpose: " << hlo->ToString();

    if (!ShapeUtil::TransposeIsBitcast(s, operand_s, hlo->dimensions())) {
      auto l0_perm = InversePermutation(ToTransposeDimensions(
          ShapeUtil::DropDegenerateDimensions(operand_s).layout()));
      auto l_perm = ToTransposeDimensions(
          ShapeUtil::DropDegenerateDimensions(s).layout());

      auto dims = NoDegenerateDims(hlo->dimensions(), s, operand_s);
      auto t = ComposePermutations(l0_perm, dims);
      auto dimensions = ComposePermutations(t, l_perm);
      auto normalized_transpose = hlo->AddInstruction(
          HloInstruction::CreateTranspose(normalized_shape, a0, dimensions));
      VLOG(3) << "Generated normalized physical transpose: "
              << normalized_transpose->ToString();
      auto bc_to_orig = MakeBitcastHlo(normalized_transpose, s);
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    } else {
      auto bc_to_orig = MakeBitcastHlo(a0, s, &hlo->metadata());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    }
    return OkStatus();
  }

  // Converts a purely physical copy into a physical+logical transposition.
  //
  // Converts:
  //
  //  A{I} -> bitcast{L} -> copy[S]{L'}
  //
  // Into:
  //
  //  A{I} -> transpose[S']{I} -> bitcast[S]{L'}
  //
  // Where S' is normalization of [S]{L'}, and transposition dimensions are
  // given by L'.
  Status HandleCopy(HloInstruction* hlo) override {
    VLOG(3) << "Processing copy: " << hlo->ToString();
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto s_normalized = Normalize(s);
    auto l0_perm = InversePermutation(ToTransposeDimensions(
        ShapeUtil::DropDegenerateDimensions(operand->shape()).layout()));
    auto l_perm =
        ToTransposeDimensions(ShapeUtil::DropDegenerateDimensions(s).layout());
    auto dimensions = ComposePermutations(l0_perm, l_perm);
    auto t = hlo->AddInstruction(
        HloInstruction::CreateTranspose(s_normalized, a0, dimensions));
    auto bc_to_orig = MakeBitcastHlo(t, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // The reverse HLO has a list of dimensions it reverses, which again becomes
  // pretty interesting in the presence of degenerate dimensions: we need to
  // drop those from the list.
  //
  // Luckily, reverse is layout-preserving.
  Status HandleReverse(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto s_normalized = Normalize(s);
    auto normalized_w_degen =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(s);

    std::vector<int64_t> new_dimensions =
        TransformDimensionsForLayoutPreservingHlo(hlo, normalized_w_degen,
                                                  s_normalized);
    auto normalized_reverse = hlo->AddInstruction(
        HloInstruction::CreateReverse(a0->shape(), a0, new_dimensions));
    auto bc_to_orig = MakeBitcastHlo(normalized_reverse, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // Padding is layout-preserving, so we only have to permute values inside the
  // padding config.
  //
  // Like in broadcast, we have to be mindful that we can't remove degenerate
  // dimensions if they are padded.
  Status HandlePad(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    const auto& operand_s = operand->shape();
    auto padded_by = hlo->mutable_operand(1);
    auto padded_config = hlo->padding_config();

    auto dim_filter = [&](int64_t dim) {
      return operand_s.dimensions(dim) != 1 ||
             !IsZeroPadding(hlo->padding_config().dimensions(dim));
    };

    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_operand,
        BitcastToNormalizedWithDegenIfNecessary(operand, dim_filter));

    auto s_normalized = Normalize(s);
    auto l = ToTransposeDimensions(s.layout());

    auto normalized_w_degen =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(s);

    PaddingConfig new_padding;
    new_padding.mutable_dimensions()->Reserve(s_normalized.dimensions_size());
    for (int dim = 0; dim < s_normalized.dimensions_size(); dim++) {
      new_padding.add_dimensions();
    }

    for (int dim = 0; dim < s.dimensions_size(); dim++) {
      if (s.dimensions(dim) == 1) {
        continue;
      }
      int tr_dim = static_cast<int>(FindIndex(l, dim));
      int out_dim = tr_dim - DegenDimsUpTo(normalized_w_degen, tr_dim);
      *new_padding.mutable_dimensions(out_dim) = padded_config.dimensions(dim);
    }

    auto padded_normalized = hlo->AddInstruction(HloInstruction::CreatePad(
        s_normalized, new_operand, padded_by, new_padding));
    auto bc_to_orig = MakeBitcastHlo(padded_normalized, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  Status HandleCustomCall(HloInstruction* hlo) override {
    if (custom_call_transformer_) {
      TF_ASSIGN_OR_RETURN(
          std::optional<HloInstruction*> transformed_custom_call,
          custom_call_transformer_(Cast<HloCustomCallInstruction>(hlo)));
      if (transformed_custom_call) {
        TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, *transformed_custom_call));
        return OkStatus();
      }
    }
    return DefaultAction(hlo);
  }

  // Pushes down bitcast across the ternary select operation: same logic as
  // HandleElementwiseBinary.
  Status HandleSelect(HloInstruction* hlo) override {
    return HandleTernary(hlo);
  }

  // DyanmicSlice is layout-preserving, so handling is analoguous to elementwise
  // unary, and transposing the elements inside the metadata, as well as the
  // operands specifying dimension sizes.
  Status HandleDynamicSlice(HloInstruction* hlo) override {
    const Shape& s = hlo->shape();
    HloInstruction* operand = hlo->mutable_operand(0);
    const Shape& operand_shape = operand->shape();
    TF_RET_CHECK(s.layout() == operand_shape.layout());

    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));

    Shape normalized_w_degen =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            operand_shape);

    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());

    std::vector<HloInstruction*> new_start_indices =
        FindNonDegenerateStartIdxs(hlo, /*param_offset=*/1, operand_shape);

    auto normalize_slice_attr = [&](absl::Span<int64_t const> input) {
      return PermuteSliceAttributes(input, layout_as_permutation,
                                    normalized_w_degen);
    };
    TF_ASSIGN_OR_RETURN(
        HloInstruction * normalized_dynamic_slice,
        MakeDynamicSliceHlo(normalized_input, new_start_indices,
                            normalize_slice_attr(hlo->dynamic_slice_sizes()),
                            &hlo->metadata()));
    *normalized_dynamic_slice->mutable_shape()->mutable_layout() =
        normalized_input->shape().layout();
    Shape normalized_shape = Normalize(s);
    // Output of slice might contain degenerate dimensions.
    HloInstruction* bc_to_normalized =
        MakeBitcastHlo(normalized_dynamic_slice, normalized_shape);
    HloInstruction* bc_to_orig = MakeBitcastHlo(bc_to_normalized, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  Status HandleDynamicUpdateSlice(HloInstruction* hlo) override {
    const Shape& s = hlo->shape();
    HloInstruction* operand = hlo->mutable_operand(0);

    HloInstruction* update = hlo->mutable_operand(1);
    const Shape& operand_shape = operand->shape();
    TF_RET_CHECK(s.layout() == operand_shape.layout());

    auto shape_filter = [&](int64_t dim) {
      return operand->shape().dimensions(dim) != 1;
    };

    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_operand,
        BitcastToNormalizedWithDegenIfNecessary(operand, shape_filter));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_update,
        BitcastToNormalizedWithDegenIfNecessary(update, shape_filter));
    std::vector<HloInstruction*> new_start_indices =
        FindNonDegenerateStartIdxs(hlo, /*param_offset=*/2, operand_shape);

    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_dus,
        MakeDynamicUpdateSliceHlo(new_operand, new_update, new_start_indices,
                                  &hlo->metadata()));
    *new_dus->mutable_shape()->mutable_layout() = new_operand->shape().layout();

    // Output of DUS might contain degenerate dimensions.
    Shape normalized_shape = Normalize(s);
    HloInstruction* bc_to_normalized =
        MakeBitcastHlo(new_dus, normalized_shape);
    HloInstruction* bc_to_orig = MakeBitcastHlo(bc_to_normalized, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));

    return OkStatus();
  }

  Status HandleClamp(HloInstruction* hlo) override {
    return HandleTernary(hlo);
  }

 private:
  // Replace clamp/select ternary operation with a normalized one.
  Status HandleTernary(HloInstruction* hlo) {
    Shape s = hlo->shape();
    HloOpcode opcode = hlo->opcode();
    TF_RET_CHECK(opcode == HloOpcode::kClamp || opcode == HloOpcode::kSelect);
    HloInstruction* p = hlo->mutable_operand(0);
    HloInstruction* i1 = hlo->mutable_operand(1);
    HloInstruction* i2 = hlo->mutable_operand(2);
    TF_RET_CHECK(p->shape().layout() == s.layout());
    TF_RET_CHECK(i1->shape().layout() == s.layout());
    TF_RET_CHECK(i2->shape().layout() == s.layout());

    TF_ASSIGN_OR_RETURN(HloInstruction * p_0, GetNormalizedInput(p));
    TF_ASSIGN_OR_RETURN(HloInstruction * i1_0, GetNormalizedInput(i1));
    TF_ASSIGN_OR_RETURN(HloInstruction * i2_0, GetNormalizedInput(i2));

    TF_ASSIGN_OR_RETURN(Shape new_shape, ShapeInference::InferTernaryOpShape(
                                             opcode, p_0, i1_0, i2_0));
    HloInstruction* normalized = hlo->parent()->AddInstruction(
        HloInstruction::CreateTernary(new_shape, opcode, p_0, i1_0, i2_0));
    hlo->SetupDerivedInstruction(normalized);

    HloInstruction* bc_to_orig = MakeBitcastHlo(normalized, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  std::vector<HloInstruction*> FindNonDegenerateStartIdxs(
      HloInstruction* hlo, int param_offset, const Shape& operand_shape) {
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(operand_shape.layout());
    std::vector<HloInstruction*> start_indices;
    for (int i = param_offset; i < hlo->operand_count(); i++) {
      start_indices.push_back(hlo->mutable_operand(i));
    }
    Shape normalized_w_degen =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            operand_shape);

    std::vector<HloInstruction*> permuted_start_indices =
        Permute(start_indices, layout_as_permutation);
    std::vector<HloInstruction*> new_start_indices;
    for (int i = 0; i < permuted_start_indices.size(); i++) {
      if (normalized_w_degen.dimensions(i) != 1) {
        new_start_indices.push_back(permuted_start_indices[i]);
      }
    }
    return new_start_indices;
  }

  StatusOr<HloInstruction*> BitcastToNormalizedWithDegenIfNecessary(
      HloInstruction* operand,
      absl::FunctionRef<bool(int64_t)> keep_dimension) {
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_operand,
                        GetNormalizedInput(operand));
    Shape operand_shape_filtered =
        ShapeUtil::FilterDimensions(keep_dimension, operand->shape());
    Shape operand_shape_normalized =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            operand_shape_filtered);
    return operand_shape_normalized == normalized_operand->shape()
               ? normalized_operand
               : MakeBitcastHlo(normalized_operand, operand_shape_normalized);
  }

  std::vector<int64_t> PermuteSliceAttributes(
      absl::Span<int64_t const> input,
      absl::Span<int64_t const> layout_as_permutation,
      const Shape& normalized_operand_w_degen) {
    std::vector<int64_t> v = Permute(input, layout_as_permutation);
    std::vector<int64_t> out;
    // Slicing on degenerate dimensions only produces degenerate dimensions,
    // so these can be safely ignored.
    for (int i = 0; i < v.size(); i++) {
      if (normalized_operand_w_degen.dimensions(i) != 1) {
        out.push_back(v[i]);
      }
    }
    return out;
  }

  bool IsZeroPadding(const PaddingConfig::PaddingConfigDimension& c) {
    return c.edge_padding_high() == 0 && c.edge_padding_low() == 0 &&
           c.interior_padding() == 0;
  }

  // Returns a list of dimensions associated with `hlo` after layout
  // normalization.
  std::vector<int64_t> TransformDimensionsForLayoutPreservingHlo(
      HloInstruction* hlo, const Shape& normalized_shape_w_degen,
      const Shape& normalized_out_shape) {
    bool skip_degen_dims = normalized_shape_w_degen != normalized_out_shape;
    std::vector<int64_t> new_dimensions;
    const auto& s = hlo->shape();
    auto l = ToTransposeDimensions(s.layout());

    for (int64_t dim : hlo->dimensions()) {
      if (s.dimensions(dim) == 1 && skip_degen_dims) {
        continue;
      }

      auto tr_dim = FindIndex(l, dim);
      auto degen_delta =
          skip_degen_dims ? DegenDimsUpTo(normalized_shape_w_degen, tr_dim) : 0;
      new_dimensions.push_back(tr_dim - degen_delta);
    }
    absl::c_sort(new_dimensions);
    return new_dimensions;
  }

  // Returns number of degenerate dimensions in `shape` up to (exclusive) a
  // `dim`.
  int DegenDimsUpTo(const Shape& shape, int dim) {
    return absl::c_count_if(shape.dimensions().subspan(0, dim),
                            [&](int d) { return d == 1; });
  }

  // Drops items from `dimensions` corresponding to degenerate dimensions in
  // `input_shape`.
  std::vector<int64_t> NoDegenerateDims(absl::Span<int64_t const> dimensions,
                                        const Shape& input_shape,
                                        const Shape& output_shape) {
    std::vector<int64_t> out;
    for (int i = 0; i < dimensions.size(); i++) {
      if (input_shape.dimensions(i) != 1) {
        int64_t val = dimensions[i];

        // Count all preceding 1-sized dimensions.
        int64_t delta = 0;
        for (int o = 0; o < val; o++) {
          if (output_shape.dimensions(o) == static_cast<int64_t>(1)) {
            delta++;
          }
        }

        out.push_back(val - delta);
      }
    }
    return out;
  }

  // Converts a layout to a dimensions transposition necessary to get to that
  // layout from identity.
  std::vector<int64_t> ToTransposeDimensions(const Layout& l) {
    std::vector<int64_t> out(l.minor_to_major().begin(),
                             l.minor_to_major().end());
    absl::c_reverse(out);
    return out;
  }

  // Due to Local Precondition we have, the input to all processed ops should
  // be HLO in descending layout piped through bitcast.
  StatusOr<HloInstruction*> GetNormalizedInput(HloInstruction* hlo) {
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kBitcast)
        << "Unexpected HLO input: " << hlo->ToString();
    auto input = hlo->mutable_operand(0);
    auto input_shape = input->shape();
    TF_RET_CHECK(input_shape.layout() ==
                 LayoutUtil::GetDefaultLayoutForShape(input_shape));
    return input;
  }

  // Forces the layout to be descending and removes degenerate dimensions
  // without altering physical layout.
  Shape Normalize(const Shape& s) {
    return ShapeUtil::DropDegenerateDimensions(
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(s));
  }

  CustomCallTransformer custom_call_transformer_;
};

}  // end namespace

StatusOr<bool> LayoutNormalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return LayoutNormalizationVisitor{custom_call_transformer_}.RunOnModule(
      module, execution_threads);
}

}  // end namespace xla
