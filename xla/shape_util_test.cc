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

#include "xla/shape_util.h"

#include <numeric>
#include <optional>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/status_macros.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(ShapeUtilTest, GetDimensionHelperCanNegativeIndex) {
  Shape matrix = ShapeUtil::MakeShape(F32, {2, 3});
  EXPECT_EQ(3, ShapeUtil::GetDimension(matrix, -1));
  EXPECT_EQ(2, ShapeUtil::GetDimension(matrix, -2));
}

TEST(ShapeUtilTest, GetDimensionHelperExampleInDocumentationTest) {
  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 3, 4});
  ASSERT_EQ(4, ShapeUtil::GetDimension(shape, -1));
}

TEST(ShapeUtilTest, NegativeIndexOobFails) {
  Shape matrix = ShapeUtil::MakeShape(F32, {2, 3});
  ASSERT_DEATH(ShapeUtil::GetDimension(matrix, -3), "dimension_number >= 0");
}

TEST(ShapeUtilTest, Rank1DimensionIndexing) {
  Shape shape = ShapeUtil::MakeShape(F32, {3});
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, Rank2DimensionIndexing) {
  Shape shape = ShapeUtil::MakeShape(F32, {3, 2});
  ASSERT_EQ(2, shape.dimensions(1));
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, Rank3DimensionIndexing) {
  Shape shape = ShapeUtil::MakeShape(F32, {3, 2, 7});
  ASSERT_EQ(7, shape.dimensions(2));
  ASSERT_EQ(2, shape.dimensions(1));
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, Rank4DimensionIndexing) {
  Shape shape = ShapeUtil::MakeShape(F32, {3, 2, 7, 8});
  ASSERT_EQ(8, shape.dimensions(3));
  ASSERT_EQ(7, shape.dimensions(2));
  ASSERT_EQ(2, shape.dimensions(1));
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, CompatibleIdenticalShapes) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {3, 2});
  Shape shape2 = ShapeUtil::MakeShape(F32, {3, 2});
  ASSERT_TRUE(ShapeUtil::Compatible(shape1, shape2));
}

TEST(ShapeUtilTest, TokenCompatibility) {
  EXPECT_TRUE(ShapeUtil::Compatible(ShapeUtil::MakeTokenShape(),
                                    ShapeUtil::MakeTokenShape()));
  EXPECT_FALSE(ShapeUtil::Compatible(ShapeUtil::MakeTokenShape(),
                                     ShapeUtil::MakeShape(F32, {})));
  EXPECT_FALSE(ShapeUtil::Compatible(ShapeUtil::MakeShape(F32, {}),
                                     ShapeUtil::MakeTokenShape()));
  EXPECT_TRUE(ShapeUtil::Compatible(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeTokenShape()}),
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeTokenShape()})));
}

TEST(ShapeUtilTest, TokensEqualShapes) {
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeTokenShape(),
                               ShapeUtil::MakeTokenShape()));
  EXPECT_FALSE(ShapeUtil::Equal(ShapeUtil::MakeTokenShape(),
                                ShapeUtil::MakeShape(F32, {})));
  EXPECT_FALSE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {}),
                                ShapeUtil::MakeTokenShape()));
  EXPECT_TRUE(ShapeUtil::Equal(
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTokenShape(),
           ShapeUtil::MakeShapeWithDenseLayout(S32, {3, 4}, {0, 1})}),
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTokenShape(),
           ShapeUtil::MakeShapeWithDenseLayout(S32, {3, 4}, {0, 1})})));
  EXPECT_FALSE(ShapeUtil::Equal(
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTokenShape(),
           ShapeUtil::MakeShapeWithDenseLayout(S32, {3, 4}, {0, 1})}),
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTokenShape(),
           ShapeUtil::MakeShapeWithDenseLayout(S32, {3, 4}, {1, 0})})));
}

TEST(ShapeUtilTest, CompatibleNotIdenticalShapes) {
  Shape shape_1 = ShapeUtil::MakeShape(F32, {3, 2});
  auto layout_1 = shape_1.mutable_layout();
  layout_1->clear_minor_to_major();
  layout_1->add_minor_to_major(0);
  layout_1->add_minor_to_major(1);

  Shape shape_2 = ShapeUtil::MakeShape(F32, {3, 2});
  auto layout_2 = shape_2.mutable_layout();
  layout_2->clear_minor_to_major();
  layout_2->add_minor_to_major(1);
  layout_2->add_minor_to_major(0);

  EXPECT_FALSE(ShapeUtil::Equal(shape_1, shape_2));
  EXPECT_TRUE(ShapeUtil::Compatible(shape_1, shape_2));
}

TEST(ShapeUtilTest, CompatibleIgnoringFpPrecision) {
  Shape shape1 = ShapeUtil::MakeShape(BF16, {3, 2});
  Shape shape2 = ShapeUtil::MakeShape(F32, {3, 2});
  ASSERT_TRUE(ShapeUtil::CompatibleIgnoringFpPrecision(shape1, shape2));
}

TEST(ShapeUtilTest, IncompatibleIgnoringFpPrecision) {
  Shape shape1 = ShapeUtil::MakeShape(BF16, {3, 2});
  Shape shape2 = ShapeUtil::MakeShape(F32, {2, 2});
  ASSERT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(shape1, shape2));
}

TEST(ShapeUtilTest, IncompatibleDifferentElementShapes) {
  Shape shape_1 = ShapeUtil::MakeShape(F32, {3, 2});
  Shape shape_2 = ShapeUtil::MakeShape(PRED, {3, 2});
  EXPECT_FALSE(ShapeUtil::Compatible(shape_1, shape_2));
}

TEST(ShapeUtilTest, EqualIgnoringFpPrecision) {
  EXPECT_TRUE(ShapeUtil::EqualIgnoringFpPrecision(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {4, 3}, {0, 1})));
}

TEST(ShapeUtilTest, UnequalIgnoringFpPrecision) {
  EXPECT_FALSE(ShapeUtil::EqualIgnoringFpPrecision(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {3, 4}, {0, 1})));
  EXPECT_FALSE(ShapeUtil::EqualIgnoringFpPrecision(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 4}, {0, 1}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {3, 4}, {1, 0})));
  EXPECT_FALSE(ShapeUtil::EqualIgnoringFpPrecision(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithDenseLayout(PRED, {4, 3}, {0, 1})));
}

TEST(ShapeUtilTest, EqualIgnoringElementType) {
  EXPECT_TRUE(ShapeUtil::EqualIgnoringElementType(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {4, 3}, {0, 1})));
  EXPECT_TRUE(ShapeUtil::EqualIgnoringElementType(
      ShapeUtil::MakeShapeWithDenseLayout(S32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {4, 3}, {0, 1})));
  EXPECT_TRUE(ShapeUtil::EqualIgnoringElementType(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithDenseLayout(PRED, {4, 3}, {0, 1})));
}

TEST(ShapeUtilTest, UnequalIgnoringElementType) {
  EXPECT_FALSE(ShapeUtil::EqualIgnoringElementType(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {3, 4}, {0, 1})));
  EXPECT_FALSE(ShapeUtil::EqualIgnoringElementType(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 4}, {0, 1}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {3, 4}, {1, 0})));
}

TEST(ShapeUtilTest, EqualDynamicShapes) {
  EXPECT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {4, 3}, {true, false}),
                       ShapeUtil::MakeShape(F32, {4, 3}, {true, false})));
  EXPECT_FALSE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {4, 3}, {true, false}),
                       ShapeUtil::MakeShape(F32, {4, 3}, {false, false})));
}

TEST(ShapeUtilTest, CompatibleDynamicShapes) {
  Shape shape_a = ShapeUtil::MakeShape(F32, {4, 3}, {true, false});
  *shape_a.mutable_layout() = Layout({1, 0});
  Shape shape_b = ShapeUtil::MakeShape(F32, {4, 3}, {true, false});
  *shape_b.mutable_layout() = Layout({0, 1});
  Shape shape_c = ShapeUtil::MakeShape(F32, {4, 3}, {false, true});
  *shape_c.mutable_layout() = Layout({0, 1});

  EXPECT_TRUE(ShapeUtil::Compatible(shape_a, shape_a));
  EXPECT_TRUE(ShapeUtil::Compatible(shape_a, shape_b));
  EXPECT_TRUE(ShapeUtil::Compatible(shape_a, shape_c));
}

TEST(ShapeUtilTest, CompatibleTuples) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(PRED, {4, 5})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(PRED, {4, 5})});
  EXPECT_TRUE(ShapeUtil::Compatible(tuple1, tuple2));
}

TEST(ShapeUtilTest, MakeMaybeTupleShape) {
  Shape s1 =
      ShapeUtil::MakeMaybeTupleShape({ShapeUtil::MakeShape(F32, {3, 2})});
  EXPECT_TRUE(ShapeUtil::Compatible(s1, ShapeUtil::MakeShape(F32, {3, 2})));
}

TEST(ShapeUtilTest, CompatibleTuplesIgnoringFpPrecision) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(BF16, {3, 2}), ShapeUtil::MakeShape(F32, {4, 5})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F64, {3, 2}), ShapeUtil::MakeShape(BF16, {4, 5})});
  EXPECT_TRUE(ShapeUtil::CompatibleIgnoringFpPrecision(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesWithSwappedElements) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(PRED, {4, 5})});
  EXPECT_FALSE(ShapeUtil::Compatible(tuple1, tuple2));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringElementType(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesIgnoringFpPrecision) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(BF16, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(BF16, {4, 5})});
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesWithDifferentPrimitiveType) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(S32, {3, 2})});
  EXPECT_FALSE(ShapeUtil::Compatible(tuple1, tuple2));
  EXPECT_TRUE(ShapeUtil::CompatibleIgnoringElementType(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesWithDifferentDimensions) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {4, 2})});
  EXPECT_FALSE(ShapeUtil::Compatible(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleScalarVsTuple) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {});
  Shape shape2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(U32, {})});
  EXPECT_FALSE(ShapeUtil::Compatible(shape1, shape2));
  EXPECT_FALSE(ShapeUtil::Compatible(shape2, shape1));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringElementType(shape1, shape2));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringElementType(shape2, shape1));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(shape1, shape2));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(shape2, shape1));
}

TEST(ShapeUtilTest, OpaqueVsArray) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {5, 7});
  Shape shape2 = ShapeUtil::MakeOpaqueShape();
  EXPECT_FALSE(ShapeUtil::Compatible(shape1, shape2));
  EXPECT_FALSE(ShapeUtil::Compatible(shape2, shape1));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(shape1, shape2));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(shape2, shape1));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringElementType(shape1, shape2));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringElementType(shape2, shape1));
}

TEST(ShapeUtilTest, ScalarDefaultLayoutEqualsScalarEmptyMin2Maj) {
  Shape scalar_default_layout = ShapeUtil::MakeShape(F32, {});
  ASSERT_TRUE(scalar_default_layout.has_layout())
      << ShapeUtil::HumanStringWithLayout(scalar_default_layout);

  const Shape scalar_empty_min2maj =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {}, {});
  ASSERT_TRUE(scalar_empty_min2maj.has_layout())
      << ShapeUtil::HumanStringWithLayout(scalar_empty_min2maj);

  EXPECT_TRUE(ShapeUtil::Equal(scalar_default_layout, scalar_empty_min2maj));
}

TEST(ShapeUtilTest, ByteSizeOfWithoutPadding) {
  EXPECT_EQ(4, ShapeUtil::ByteSizeOfPrimitiveType(F32));
  EXPECT_EQ(4, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F32, {})));
  EXPECT_EQ(800, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F32, {10, 20})));

  EXPECT_EQ(8, ShapeUtil::ByteSizeOfPrimitiveType(F64));
  EXPECT_EQ(8, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F64, {})));
  EXPECT_EQ(1600, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F64, {10, 20})));

  EXPECT_EQ(8, ShapeUtil::ByteSizeOfPrimitiveType(C64));
  EXPECT_EQ(8, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(C64, {})));
  EXPECT_EQ(1600, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(C64, {10, 20})));

  EXPECT_EQ(0, ShapeUtil::ByteSizeOfPrimitiveType(TOKEN));
  EXPECT_EQ(0, ShapeUtil::ByteSizeOf(ShapeUtil::MakeTokenShape()));
}

TEST(ShapeUtilTest, NilShape) {
  EXPECT_TRUE(ShapeUtil::IsEmptyTuple(ShapeUtil::MakeNil()));
  EXPECT_FALSE(ShapeUtil::IsEmptyTuple(ShapeUtil::MakeShape(F32, {1, 2, 3})));
  EXPECT_FALSE(ShapeUtil::IsEmptyTuple(ShapeUtil::MakeShape(F32, {0, 1})));
  EXPECT_FALSE(ShapeUtil::IsEmptyTuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {})})));
  EXPECT_FALSE(ShapeUtil::IsEmptyTuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {0})})));
}

TEST(ShapeUtilTest, NestedTuple) {
  EXPECT_FALSE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape({})));
  EXPECT_FALSE(ShapeUtil::IsNestedTuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {})})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeTupleShape({})})));
  EXPECT_FALSE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(S32, {})})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeTupleShape({})})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({}), ShapeUtil::MakeShape(S32, {})})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({}), ShapeUtil::MakeTupleShape({})})));
}

TEST(ShapeUtilTest, NestedTupleWithPtrs) {
  const Shape nil = ShapeUtil::MakeNil();
  const Shape s32 = ShapeUtil::MakeShape(S32, {});
  EXPECT_FALSE(ShapeUtil::IsNestedTuple(nil));
  EXPECT_FALSE(
      ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShapeWithPtrs({&s32})));
  EXPECT_TRUE(
      ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShapeWithPtrs({&nil})));
  EXPECT_FALSE(ShapeUtil::IsNestedTuple(
      ShapeUtil::MakeTupleShapeWithPtrs({&s32, &s32})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(
      ShapeUtil::MakeTupleShapeWithPtrs({&s32, &nil})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(
      ShapeUtil::MakeTupleShapeWithPtrs({&nil, &s32})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(
      ShapeUtil::MakeTupleShapeWithPtrs({&nil, &nil})));
}

TEST(ShapeUtilTest, ElementsIn) {
  EXPECT_EQ(1, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {})));
  EXPECT_EQ(0, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {0})));
  EXPECT_EQ(1, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {1})));
  EXPECT_EQ(1, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {1, 1})));
  EXPECT_EQ(2, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {2})));
  EXPECT_EQ(2, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {2, 1})));
  EXPECT_EQ(15, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {3, 5})));
  EXPECT_EQ(0, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {3, 0, 5})));
  EXPECT_EQ(0, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {0, 3, 0})));
  EXPECT_EQ(15, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {1, 3, 5})));
  EXPECT_EQ(221, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {13, 17})));
}

TEST(ShapeUtilTest, HasPrimitiveType) {
  EXPECT_TRUE(ShapeUtil::HasPrimitiveType(ShapeUtil::MakeShape(S32, {}), S32));
  EXPECT_FALSE(ShapeUtil::HasPrimitiveType(ShapeUtil::MakeShape(S32, {}), S16));
  EXPECT_TRUE(ShapeUtil::HasPrimitiveType(ShapeUtil::MakeShape(S32, {0}), S32));
  EXPECT_FALSE(ShapeUtil::HasPrimitiveType(ShapeUtil::MakeTupleShape({}), S32));
  EXPECT_TRUE(ShapeUtil::HasPrimitiveType(
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(S32, {})}),
      S32));
  EXPECT_TRUE(ShapeUtil::HasPrimitiveType(
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(S32, {}),
           ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S16, {})})}),
      S16));
}

TEST(ShapeUtilTest, IsZeroElementArray) {
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {})));
  EXPECT_TRUE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {0})));
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {1})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {1, 1})));
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {2})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {2, 1})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {3, 5})));
  EXPECT_TRUE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {3, 0, 5})));
  EXPECT_TRUE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {0, 3, 0})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {1, 3, 5})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {13, 17})));

  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeNil()));
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeTupleShape({})));
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {0, 3, 0})})));
}

TEST(ShapeUtilTest, SameDimensions) {
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {}),
                                        ShapeUtil::MakeShape(S32, {})));
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {}),
                                        ShapeUtil::MakeShape(F32, {})));
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                        ShapeUtil::MakeShape(S32, {1})));
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {0}),
                                        ShapeUtil::MakeShape(S32, {0})));
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {2}),
                                        ShapeUtil::MakeShape(S32, {2})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                         ShapeUtil::MakeShape(F32, {2})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {0, 0}),
                                         ShapeUtil::MakeShape(F32, {0})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                         ShapeUtil::MakeShape(F32, {1, 1})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {}),
                                         ShapeUtil::MakeShape(F32, {1})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                         ShapeUtil::MakeShape(F32, {1, 1})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                         ShapeUtil::MakeShape(F32, {1, 0})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1, 1}),
                                         ShapeUtil::MakeShape(F32, {1, 2})));
}

TEST(ShapeUtilTest, GetSubshape) {
  // Test array shape.
  Shape array_shape = ShapeUtil::MakeShape(F32, {42, 42, 123});
  EXPECT_TRUE(
      ShapeUtil::Equal(array_shape, ShapeUtil::GetSubshape(array_shape, {})));
  EXPECT_TRUE(ShapeUtil::Equal(
      array_shape, *ShapeUtil::GetMutableSubshape(&array_shape, {})));

  // Test tuple shape.
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({array_shape, array_shape, array_shape});
  EXPECT_TRUE(
      ShapeUtil::Equal(tuple_shape, ShapeUtil::GetSubshape(tuple_shape, {})));
  EXPECT_TRUE(
      ShapeUtil::Equal(array_shape, ShapeUtil::GetSubshape(tuple_shape, {0})));
  EXPECT_TRUE(
      ShapeUtil::Equal(array_shape, ShapeUtil::GetSubshape(tuple_shape, {1})));
  EXPECT_TRUE(
      ShapeUtil::Equal(array_shape, ShapeUtil::GetSubshape(tuple_shape, {2})));

  // Test nested tuple shape.
  Shape nested_tuple_shape = ShapeUtil::MakeTupleShape(
      {array_shape, ShapeUtil::MakeTupleShape({array_shape, array_shape}),
       ShapeUtil::MakeTupleShape(
           {ShapeUtil::MakeTupleShape({array_shape, array_shape}),
            array_shape})});
  EXPECT_TRUE(ShapeUtil::Equal(nested_tuple_shape,
                               ShapeUtil::GetSubshape(nested_tuple_shape, {})));
  EXPECT_TRUE(ShapeUtil::Equal(
      array_shape, ShapeUtil::GetSubshape(nested_tuple_shape, {0})));
  EXPECT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeTupleShape({array_shape, array_shape}),
                       ShapeUtil::GetSubshape(nested_tuple_shape, {1})));
  EXPECT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeTupleShape({array_shape, array_shape}),
                       ShapeUtil::GetSubshape(nested_tuple_shape, {2, 0})));
}

TEST(ShapeUtilTest, IsLeafIndex) {
  // Test array shape.
  Shape array_shape = ShapeUtil::MakeShape(F32, {42, 42, 123});
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(array_shape, {}));

  // Test tuple shape.
  Shape tuple_shape = ShapeUtil::MakeTupleShape({array_shape, array_shape});
  EXPECT_FALSE(ShapeUtil::IsLeafIndex(tuple_shape, {}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(tuple_shape, {0}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(tuple_shape, {1}));

  // Test nested tuple shape.
  Shape nested_tuple_shape = ShapeUtil::MakeTupleShape(
      {array_shape, ShapeUtil::MakeTupleShape({array_shape, array_shape}),
       ShapeUtil::MakeTupleShape(
           {ShapeUtil::MakeTupleShape({array_shape, array_shape}),
            array_shape})});
  EXPECT_FALSE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {0}));
  EXPECT_FALSE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {1}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {1, 0}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {1, 1}));
}

TEST(ShapeUtilTest, ForEachSubshapeArray) {
  const Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  int calls = 0;
  ShapeUtil::ForEachSubshape(
      shape, [&calls, &shape](const Shape& subshape, const ShapeIndex& index) {
        EXPECT_EQ(&shape, &subshape);
        EXPECT_TRUE(index.empty());
        ++calls;
      });
  EXPECT_EQ(1, calls);
}

TEST(ShapeUtilTest, ForEachSubshapeNestedTuple) {
  const Shape shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {42}),
       ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {101}),
                                  ShapeUtil::MakeShape(PRED, {33})})});
  int calls = 0;
  ShapeUtil::ForEachSubshape(
      shape, [&calls, &shape](const Shape& subshape, const ShapeIndex& index) {
        EXPECT_TRUE(
            ShapeUtil::Equal(subshape, ShapeUtil::GetSubshape(shape, index)));
        if (calls == 0) {
          // Visitation should go from outside in.
          EXPECT_TRUE(index.empty());
        } else if (calls == 4) {
          // Last visitation should be to the array with 33 elements.
          EXPECT_EQ(33, ShapeUtil::ElementsIn(subshape));
        }
        ++calls;
      });
  EXPECT_EQ(5, calls);
}

TEST(ShapeUtilTest, ForEachMutableSubshapeNestedTuple) {
  Shape shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {42}),
       ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {101}),
                                  ShapeUtil::MakeShape(PRED, {33})})});
  int calls = 0;
  ShapeUtil::ForEachMutableSubshape(
      &shape, [&calls, &shape](const Shape* subshape, const ShapeIndex& index) {
        // Pointer values should be equal
        EXPECT_EQ(subshape, ShapeUtil::GetMutableSubshape(&shape, index));
        if (calls == 0) {
          // Visitation should go from outside in.
          EXPECT_TRUE(index.empty());
        } else if (calls == 4) {
          // Last visitation should be to the array with 33 elements.
          EXPECT_EQ(33, ShapeUtil::ElementsIn(*subshape));
        }
        ++calls;
      });
  EXPECT_EQ(5, calls);
}

TEST(ShapeUtilTest, InsertedOrDeleted1SizedDimensions) {
  Shape shape0 = ShapeUtil::MakeShape(S32, {9, 1, 4});
  Shape shape1 = ShapeUtil::MakeShape(S32, {1, 9, 4, 1});
  Shape shape2 = ShapeUtil::MakeShape(S32, {3, 1, 12});
  EXPECT_TRUE(
      ShapeUtil::InsertedOrDeleted1SizedDimensions(shape0, shape1).has_value());
  EXPECT_FALSE(
      ShapeUtil::InsertedOrDeleted1SizedDimensions(shape0, shape2).has_value());
}

TEST(ShapeUtilTest, ForEachIndex) {
  struct ShapeDimensionAndNumberInvocations {
    std::vector<int64_t> dimensions;
    int invocations;
  } test_data[] = {
      {{}, 1},     {{0}, 0},      {{16}, 16},          {{3, 0}, 0},
      {{0, 2}, 0}, {{4, 16}, 64}, {{6, 11, 17}, 1122}, {{6, 11, 5, 17}, 5610},
  };

  for (const auto& data : test_data) {
    Shape shape = ShapeUtil::MakeShape(F32, data.dimensions);
    // Increments at every invocation.
    int invocations = 0;
    auto increment_func = [&invocations](absl::Span<const int64_t> indexes) {
      invocations++;
      return true;
    };

    std::vector<int64_t> zero_base(data.dimensions.size(), 0);
    std::vector<int64_t> step(data.dimensions.size(), 1);

    ShapeUtil::ForEachIndex(shape, zero_base, data.dimensions, step,
                            increment_func);

    EXPECT_EQ(invocations, data.invocations);
  }
}

TEST(ShapeUtilTest, ForEachIndexWithStatus) {
  Shape shape = ShapeUtil::MakeShape(F32, {10, 10});
  // Increments at every invocation.
  int invocations = 0;
  auto increment_func =
      [&invocations](absl::Span<const int64_t> indexes) -> StatusOr<bool> {
    if (++invocations == 5) {
      return Unimplemented("Cannot increment beyond 5.");
    }
    return true;
  };

  Status error_status = ShapeUtil::ForEachIndexWithStatus(
      shape, /*base=*/{0, 0}, /*count=*/{10, 10}, /*incr=*/{0, 1},
      increment_func);

  EXPECT_FALSE(error_status.ok());
  EXPECT_THAT(error_status.error_message(),
              ::testing::HasSubstr("Cannot increment beyond 5."));
  EXPECT_EQ(invocations, 5);
}

TEST(ShapeUtilTest, GetForEachIndexParallelThreadCount) {
  const int kThreadCount = ShapeUtil::GetForEachIndexParallelThreadCount();

  Shape shape = ShapeUtil::MakeShape(F32, {10, 100});
  auto check_func = [kThreadCount](absl::Span<const int64_t> /*indexes*/,
                                   int thread_id) -> StatusOr<bool> {
    EXPECT_GE(thread_id, -1);
    EXPECT_LT(thread_id, kThreadCount);
    return true;
  };

  for (int i = 0; i < 10; ++i) {
    ShapeUtil::ForEachIndexParallel(shape, /*base=*/{0, 0}, /*count=*/{10, 100},
                                    /*incr=*/{1, 1}, check_func);
  }
}

TEST(ShapeUtilTest, ForEachIndexParallel) {
  Shape shape = ShapeUtil::MakeShape(F32, {10, 10});
  int64_t output[10][10];
  int init = 5;
  auto set_func = [&](absl::Span<const int64_t> indexes,
                      int /*thread_id*/) -> StatusOr<bool> {
    output[indexes[0]][indexes[1]] = init + indexes[0] + indexes[1];
    return true;
  };

  ShapeUtil::ForEachIndexParallel(shape, /*base=*/{0, 0}, /*count=*/{10, 10},
                                  /*incr=*/{1, 1}, set_func);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      EXPECT_EQ(output[i][j], init + i + j);
    }
  }
}

TEST(ShapeUtilTest, ForEachIndexParallel_Rank0) {
  Shape shape = ShapeUtil::MakeShape(F32, {});
  int64_t output = -1;
  auto set_func = [&](absl::Span<const int64_t> indexes,
                      int /*thread_id*/) -> StatusOr<bool> {
    output = indexes.size();
    return true;
  };

  ShapeUtil::ForEachIndexParallel(shape, /*base=*/{}, /*count=*/{},
                                  /*incr=*/{}, set_func);

  EXPECT_EQ(output, 0);
}

TEST(ShapeUtilTest, ForEachIndexParallel_Empty) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 0});
  bool called = false;
  auto set_func = [&](absl::Span<const int64_t> indexes,
                      int /*thread_id*/) -> StatusOr<bool> {
    called = true;
    return true;
  };

  ShapeUtil::ForEachIndexParallel(shape, /*base=*/{0, 0}, /*count=*/{2, 0},
                                  /*incr=*/{1, 1}, set_func);

  EXPECT_FALSE(called);
}

TEST(ShapeUtilTest, ForEachIndexParallel_DimensionPinnedWithZeros) {
  // Some users of ForEachIndex use base = a, count = 0, incr = 0 to indicate
  // that the given dimension should be pinned to the value "a" during the
  // iteration. We want to be compatible with this behavior so we test it here.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  int64_t output[2][2] = {};
  int init = 5;
  auto set_func = [&](absl::Span<const int64_t> indexes,
                      int /*thread_id*/) -> StatusOr<bool> {
    output[indexes[0]][indexes[1]] = init + indexes[0] + indexes[1];
    return true;
  };

  ShapeUtil::ForEachIndexParallel(shape, /*base=*/{1, 0}, /*count=*/{0, 2},
                                  /*incr=*/{0, 1}, set_func);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (i == 1) {
        EXPECT_EQ(output[i][j], init + i + j);
      } else {
        EXPECT_EQ(output[i][j], 0);
      }
    }
  }
}

TEST(ShapeUtilTest, ForEachIndexParallel_WithSkips) {
  Shape shape = ShapeUtil::MakeShape(F32, {10, 10});
  int64_t output[10][10] = {};
  int init = 5;
  auto set_func = [&](absl::Span<const int64_t> indexes,
                      int /*thread_id*/) -> StatusOr<bool> {
    output[indexes[0]][indexes[1]] = init + indexes[0] + indexes[1];
    return true;
  };

  ShapeUtil::ForEachIndexParallel(shape, /*base=*/{2, 3}, /*count=*/{3, 1},
                                  /*incr=*/{2, 1}, set_func);

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      if ((i == 2 || i == 4) && j == 3) {
        EXPECT_EQ(output[i][j], init + i + j);
      } else {
        EXPECT_EQ(output[i][j], 0);
      }
    }
  }
}

TEST(ShapeUtilTest, ForEachIndexParallel_CalledTwice) {
  Shape shape = ShapeUtil::MakeShape(F32, {10, 10});
  int64_t output[10][10];
  int init = 5;
  auto set_func = [&](absl::Span<const int64_t> indexes,
                      int /*thread_id*/) -> StatusOr<bool> {
    output[indexes[0]][indexes[1]] = init + indexes[0] + indexes[1];
    return true;
  };
  int init2 = 15;
  auto set_func2 = [&](absl::Span<const int64_t> indexes,
                       int /*thread_id*/) -> StatusOr<bool> {
    output[indexes[0]][indexes[1]] = init2 + indexes[0] + indexes[1];
    return true;
  };

  ShapeUtil::ForEachIndexParallel(shape, /*base=*/{0, 0}, /*count=*/{10, 10},
                                  /*incr=*/{1, 1}, set_func);
  ShapeUtil::ForEachIndexParallel(shape, /*base=*/{0, 0}, /*count=*/{10, 10},
                                  /*incr=*/{1, 1}, set_func2);

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      EXPECT_EQ(output[i][j], init2 + i + j);
    }
  }
}

TEST(ShapeUtilTest, ForEachIndexParallel_CalledFromMultipleThreads) {
  constexpr int kCallingThreads = 10;
  constexpr int kDim0 = 10;
  constexpr int kDim1 = 10;
  constexpr int kInit = 5;
  const Shape kShape = ShapeUtil::MakeShape(F32, {kDim0, kDim1});
  int64_t output[kCallingThreads][kDim0][kDim1];

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "foreach",
                                 kCallingThreads);
    for (int t = 0; t < kCallingThreads; ++t) {
      pool.Schedule([&output, &kShape, t] {
        auto set_func = [&output, t](absl::Span<const int64_t> indexes,
                                     int /*thread_id*/) -> StatusOr<bool> {
          output[t][indexes[0]][indexes[1]] = kInit + indexes[0] + indexes[1];
          return true;
        };

        ShapeUtil::ForEachIndexParallel(kShape, /*base=*/{0, 0},
                                        /*count=*/{kDim0, kDim1},
                                        /*incr=*/{1, 1}, set_func);
      });
    }
  }

  for (int t = 0; t < kCallingThreads; ++t) {
    for (int i = 0; i < kDim0; ++i) {
      for (int j = 0; j < kDim1; ++j) {
        EXPECT_EQ(output[t][i][j], kInit + i + j);
      }
    }
  }
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_1x1x1x1_to_1x1x1) {
  // All output dimensions should be unmodified. One of the input dimensions is
  // modified because the input rank is larger by one.
  EXPECT_THAT(ShapeUtil::DimensionsUnmodifiedByReshape(
                  ShapeUtil::MakeShape(S32, {1, 1, 1, 1}),
                  ShapeUtil::MakeShape(S32, {1, 1, 1})),
              ElementsAre(std::make_pair(0, 0), std::make_pair(1, 1),
                          std::make_pair(2, 2)));
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_1x1x1_to_1x1x1x1) {
  // All input dimensions should be unmodified. One of the output dimensions is
  // modified because the output rank is larger by one.
  EXPECT_THAT(ShapeUtil::DimensionsUnmodifiedByReshape(
                  ShapeUtil::MakeShape(S32, {1, 1, 1}),
                  ShapeUtil::MakeShape(S32, {1, 1, 1, 1})),
              ElementsAre(std::make_pair(0, 0), std::make_pair(1, 1),
                          std::make_pair(2, 2)));
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_4x1x3x5x6x7_to_2x6x1x5x1x42) {
  // The only matching dimension is the one with size 5.
  // 4, 1, 3, 5, 6, 7
  //          |
  // 2, 6, 1, 5, 1, 42
  EXPECT_THAT(ShapeUtil::DimensionsUnmodifiedByReshape(
                  ShapeUtil::MakeShape(S32, {4, 1, 3, 5, 6, 7}),
                  ShapeUtil::MakeShape(S32, {2, 6, 1, 5, 1, 42})),
              ElementsAre(std::make_pair(3, 3)));
}

TEST(ShapeUtilTest, ReshapeIsBitcast_3x4_6x2) {
  for (bool input_is_row_major : {true, false}) {
    for (bool output_is_row_major : {true, false}) {
      Layout input_layout = input_is_row_major ? LayoutUtil::MakeLayout({1, 0})
                                               : LayoutUtil::MakeLayout({0, 1});
      Layout output_layout = output_is_row_major
                                 ? LayoutUtil::MakeLayout({1, 0})
                                 : LayoutUtil::MakeLayout({0, 1});
      // Suppose the input is logically (i.e. ignoring its layout)
      //   0  1  2  3
      //   4  5  6  7
      //   8  9  10 11
      //
      // The reshape transforms the input to logically
      //   0  1
      //   2  3
      //   4  5
      //   6  7
      //   8  9
      //   10 11
      //
      // The input and the output have the same underlying data only if they
      // are both row-major.
      EXPECT_EQ(ShapeUtil::ReshapeIsBitcast(
                    ShapeUtil::MakeShapeWithDenseLayout(
                        F32, {3, 4}, input_layout.minor_to_major()),
                    ShapeUtil::MakeShapeWithDenseLayout(
                        F32, {6, 2}, output_layout.minor_to_major())),
                input_is_row_major && output_is_row_major);
    }
  }
}

TEST(ShapeUtilTest, ReshapeIsBitcast_3x2x2_6x2_Dim1IsMostMinor) {
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 2, 2}, {1, 0, 2}),
      ShapeUtil::MakeShapeWithDenseLayout(F32, {6, 2}, {0, 1})));
}

TEST(ShapeUtilTest, ReshapeIsBitcastIgnoreElementType) {
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 2, 2}, {1, 0, 2}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {6, 2}, {0, 1}),
      /*ignore_element_type=*/true));
  EXPECT_FALSE(ShapeUtil::ReshapeIsBitcast(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 2, 2}, {1, 0, 2}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {6, 2}, {0, 1}),
      /*ignore_element_type=*/false));
}

TEST(ShapeUtilTest, TransposeIsBitcastIgnoreElementType) {
  EXPECT_TRUE(ShapeUtil::TransposeIsBitcast(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 5}, {1, 0}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {5, 10}, {0, 1}), {1, 0},
      /*ignore_element_type=*/true));
  EXPECT_FALSE(ShapeUtil::TransposeIsBitcast(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 5}, {1, 0}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {5, 10}, {0, 1}), {1, 0},
      /*ignore_element_type=*/false));
}

TEST(ShapeUtilTest, IsReshapeOrTransposeBitcast) {
  EXPECT_TRUE(ShapeUtil::IsReshapeOrTransposeBitcast(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 5}, {1, 0}),
      ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 10}, {0, 1})));
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 2, 2}, {1, 0, 2}),
      ShapeUtil::MakeShapeWithDenseLayout(F16, {6, 2}, {0, 1}),
      /*ignore_element_type=*/true));
}

TEST(ShapeUtilTest, HasDegenerateDimensions) {
  EXPECT_TRUE(
      ShapeUtil::HasDegenerateDimensions(ShapeUtil::MakeShape(F32, {3, 1, 2})));
  EXPECT_TRUE(
      ShapeUtil::HasDegenerateDimensions(ShapeUtil::MakeShape(F32, {3, 1, 1})));
  EXPECT_FALSE(
      ShapeUtil::HasDegenerateDimensions(ShapeUtil::MakeShape(F32, {3, 3, 5})));
  EXPECT_FALSE(
      ShapeUtil::HasDegenerateDimensions(ShapeUtil::MakeShape(F32, {3, 0, 5})));
}

TEST(ShapeUtilTest, PermuteDimensionsLayout) {
  std::vector<int64_t> layout(3);
  std::iota(layout.begin(), layout.end(), 0);
  do {
    Shape s = ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 100, 1000}, layout);
    SCOPED_TRACE(absl::StrCat("s=", ShapeUtil::HumanString(s)));

    std::vector<int64_t> permutation(3);
    std::iota(permutation.begin(), permutation.end(), 0);
    do {
      SCOPED_TRACE(
          absl::StrCat("permutation=", absl::StrJoin(permutation, ",")));

      EXPECT_TRUE(ShapeUtil::TransposeIsBitcast(
          s, ShapeUtil::PermuteDimensions(permutation, s), permutation));
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  } while (std::next_permutation(layout.begin(), layout.end()));
}

TEST(ShapeUtilTest, UpdateDynamicDimensions) {
  Shape shape = ShapeUtil::MakeShape(F32, {10, 100, 1000});

  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape});

  ShapeUtil::UpdateDynamicDimension(&tuple_shape, {0}, 1, true);
  EXPECT_TRUE(ShapeUtil::GetSubshape(tuple_shape, {0}).is_dynamic_dimension(1));
}

TEST(ShapeUtilTest, PermuteDynamicDimensions) {
  Shape shape =
      ShapeUtil::MakeShape(F32, {10, 100, 1000},
                           /*dynamic_dimensions*/ {false, true, true});
  SCOPED_TRACE(absl::StrCat("shape=", shape.ToString()));

  std::vector<int64_t> permutation(3);
  std::iota(permutation.begin(), permutation.end(), 0);
  do {
    SCOPED_TRACE(absl::StrCat("permutation=", absl::StrJoin(permutation, ",")));

    auto permuted = ShapeUtil::PermuteDimensions(permutation, shape);
    for (int i = 0; i < shape.rank(); i++) {
      EXPECT_EQ(permuted.dimensions(i), shape.dimensions(permutation[i]));
      EXPECT_EQ(permuted.is_dynamic_dimension(i),
                shape.is_dynamic_dimension(permutation[i]));
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));
}

TEST(ShapeUtilTest, MoveDimToMajor) {
  Shape shape = ShapeUtil::MakeShape(F32, {10, 10, 10});  // implicit {2, 1, 0}
  Shape new_shape = ShapeUtil::MoveDimToMajor(shape, 0);
  EXPECT_EQ(shape, new_shape);

  new_shape = ShapeUtil::MoveDimToMajor(shape, 1);
  EXPECT_EQ(new_shape,
            ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 10, 10}, {2, 0, 1}));

  shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 10, 10}, {0, 2, 1});
  new_shape = ShapeUtil::MoveDimToMajor(shape, 0);
  EXPECT_EQ(new_shape,
            ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 10, 10}, {2, 1, 0}));

  shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {10, 10, 10}),
       ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 10, 10}, {0, 2, 1})});
  new_shape = ShapeUtil::MoveDimToMajor(shape, 0);
  EXPECT_EQ(new_shape,
            ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {10, 10, 10}),
                                       ShapeUtil::MakeShapeWithDenseLayout(
                                           F32, {10, 10, 10}, {2, 1, 0})}));
}

TEST(ShapeUtilTest, DeleteDimensions) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 3, 2}, {2, 0, 1});
  Shape new_shape = ShapeUtil::DeleteDimensions({1}, shape);
  EXPECT_EQ(new_shape,
            ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 2}, {1, 0}));
}

TEST(ShapeUtilTest, MakeShapeWithDescendingLayoutAndSamePhysicalLayout) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {128, 24, 4, 48, 48},
                                                    {2, 4, 3, 1, 0});
  Shape new_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(shape);
  EXPECT_EQ(new_shape, ShapeUtil::MakeShapeWithDenseLayout(
                           F32, {128, 24, 48, 48, 4}, {4, 3, 2, 1, 0}));
}

TEST(ShapeUtilTest, DeduceTransposeDimensionsForBitcast) {
  Shape input_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 3}, {1, 0});
  Shape output_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 5}, {0, 1});
  std::vector<int64_t> expected_permutation = {1, 0};
  EXPECT_EQ(std::make_optional(expected_permutation),
            ShapeUtil::DeduceTransposeDimensionsForBitcast(input_shape,
                                                           output_shape));
}

TEST(ShapeUtilTest, DeduceTransposeDimensionsForBitcastNegative) {
  Shape input_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 3}, {1, 0});
  Shape output_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 5}, {1, 0});
  EXPECT_EQ(std::nullopt, ShapeUtil::DeduceTransposeDimensionsForBitcast(
                              input_shape, output_shape));
}

TEST(ShapeUtilTest, DeleteDimensionsUnsorted) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 3, 2, 7, 9},
                                                    {2, 0, 1, 4, 3});
  Shape a = ShapeUtil::DeleteDimensions({1, 2, 3}, shape);
  Shape b = ShapeUtil::DeleteDimensions({3, 2, 1}, shape);
  EXPECT_EQ(a, b);
  EXPECT_EQ(a, ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 9}, {0, 1}));
}

TEST(ShapeUtilTest, B_250640044) {
  // This case failed the fuzzer; see b/250640044.
  ShapeProto proto;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(element_type: TUPLE
           tuple_shapes {
             element_type: S8
             dimensions: 137438953472
             layout {
               minor_to_major: 0
               dim_level_types: DIM_COMPRESSED
               physical_shape {
                 element_type: TUPLE
                 tuple_shapes {}
               }
             }
             is_dynamic_dimension: false
           })pb",
      &proto));
  Shape shape(proto);
  EXPECT_FALSE(ShapeUtil::ValidateShape(shape).ok());
}

TEST(ShapeUtilTest, B_251055887) {
  // This case failed the fuzzer; see b/251055887.
  ShapeProto proto;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        element_type: S8
        dimensions: 0
        dimensions: 8
        dimensions: 0
        dimensions: 0
        dimensions: 4
        dimensions: 1
        dimensions: 1
        dimensions: 6
        dimensions: 281474976710657
        dimensions: 1
        layout {
          minor_to_major: 1
          minor_to_major: 3
          minor_to_major: 0
          minor_to_major: 5
          minor_to_major: 4
          minor_to_major: 6
          minor_to_major: 8
          minor_to_major: 7
          minor_to_major: 6
          minor_to_major: 9
          physical_shape { element_type: -562 }
        })pb",
      &proto));
  Shape shape(proto);
  EXPECT_FALSE(ShapeUtil::ValidateShape(shape).ok());
}

TEST(Transpose021Test, NoTranspose) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {128, 64}, {1, 0});
  Shape transposed =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {64, 128}, {0, 1});
  EXPECT_EQ(std::nullopt, ShapeUtil::GetNormalizedTransposeShape(
                              shape, transposed, Vector3{0, 2, 1}));
}

TEST(Transpose021Test, NoTranspose2) {
  Shape shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {128, 64, 32}, {2, 1, 0});
  Shape transposed =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {32, 64, 128}, {0, 1, 2});
  EXPECT_EQ(std::nullopt, ShapeUtil::GetNormalizedTransposeShape(
                              shape, transposed, Vector3{0, 1, 2}));
}

TEST(Transpose021Test, WrongTranspose) {
  Shape input_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 32768, 16}, {2, 1, 0});
  Shape output_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 32768, 16}, {0, 1, 2});
  EXPECT_EQ(std::nullopt, ShapeUtil::GetNormalizedTransposeShape(
                              input_shape, output_shape, Vector3{0, 2, 1}));
}

TEST(Transpose021Test, WrongTranspose2) {
  Shape input_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 16}, {1, 0});
  Shape output_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 16}, {0, 1});
  EXPECT_EQ(std::nullopt, ShapeUtil::GetNormalizedTransposeShape(
                              input_shape, output_shape, Vector3{0, 1, 2}));
}

TEST(Transpose021Test, WrongTranspose3) {
  Shape input_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 16}, {1, 0});
  Shape output_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 16}, {0, 1});
  EXPECT_EQ(std::nullopt, ShapeUtil::GetNormalizedTransposeShape(
                              input_shape, output_shape, Vector3{1, 2, 0}));
}

TEST(Transpose021Test, Simple) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {128, 64}, {1, 0});
  Shape transposed =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {128, 64}, {0, 1});
  EXPECT_EQ(std::make_optional(Vector3{1, 64, 128}),
            ShapeUtil::GetNormalizedTransposeShape(shape, transposed,
                                                   Vector3{0, 2, 1}));
}

TEST(Transpose021Test, Simple2) {
  Shape input_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 32768, 16}, {2, 1, 0});
  Shape output_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 32768, 16}, {1, 2, 0});
  EXPECT_EQ(std::make_optional(Vector3{8, 16, 32768}),
            ShapeUtil::GetNormalizedTransposeShape(input_shape, output_shape,
                                                   Vector3{0, 2, 1}));
}

TEST(Transpose021Test, Simple3) {
  Shape input_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 32768, 16}, {2, 1, 0});
  Shape output_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 32768, 16}, {0, 1, 2});
  EXPECT_EQ(std::make_optional(Vector3{16, 32768, 8}),
            ShapeUtil::GetNormalizedTransposeShape(input_shape, output_shape,
                                                   Vector3{2, 1, 0}));
}

TEST(Transpose021Test, Simple4) {
  Shape input_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 16}, {1, 0});
  Shape output_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 16}, {0, 1});
  EXPECT_EQ(std::make_optional(Vector3{16, 1, 8}),
            ShapeUtil::GetNormalizedTransposeShape(input_shape, output_shape,
                                                   Vector3{2, 1, 0}));
}

TEST(Transpose021Test, LargeView) {
  Shape input_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {8, 32, 32, 32, 16}, {4, 3, 2, 1, 0});
  Shape output_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {8, 32, 32, 32, 16}, {3, 2, 1, 4, 0});
  EXPECT_EQ(std::make_optional(Vector3{8, 16, 32768}),
            ShapeUtil::GetNormalizedTransposeShape(input_shape, output_shape,
                                                   Vector3{0, 2, 1}));
}

TEST(Transpose021Test, LargeSizeOverflowTest) {
  Shape input_shape =
      ShapeUtil::MakeShapeWithDenseLayout(BF16, {4096, 4096, 128}, {2, 1, 0});
  Shape output_shape =
      ShapeUtil::MakeShapeWithDenseLayout(BF16, {4096, 4096, 128}, {2, 1, 0});
  EXPECT_EQ(std::nullopt, ShapeUtil::GetNormalizedTransposeShape(
                              input_shape, output_shape, Vector3{0, 2, 1}));
}

TEST(Transpose021Test, Batched) {
  Shape shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {32, 3, 64}, {2, 1, 0});
  Shape transposed =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {32, 3, 64}, {1, 0, 2});
  EXPECT_EQ(std::make_optional(Vector3{1, 64, 96}),
            ShapeUtil::GetNormalizedTransposeShape(shape, transposed,
                                                   Vector3{0, 2, 1}));
}

TEST(Transpose021Test, BatchedLogical) {
  Shape shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {32, 3, 64}, {2, 1, 0});
  Shape transposed =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {64, 32, 3}, {2, 1, 0});
  std::vector<int64_t> dimensions = {2, 0, 1};
  EXPECT_EQ(std::make_optional(Vector3{1, 64, 96}),
            ShapeUtil::GetNormalizedLogicalTransposeShape(
                shape, transposed, dimensions, Vector3{0, 2, 1}));
}

TEST(Transpose021Test, Large) {
  Shape shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 31, 31, 65}, {3, 2, 1, 0});
  Shape transposed =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 31, 31, 65}, {2, 1, 3, 0});
  EXPECT_EQ(std::make_optional(Vector3{8, 65, 961}),
            ShapeUtil::GetNormalizedTransposeShape(shape, transposed,
                                                   Vector3{0, 2, 1}));
}

TEST(AlgebraicSimplifierTest, ReshapeIsBitcast_3x2x2_6x2_Dim0IsMostMinor) {
  EXPECT_FALSE(ShapeUtil::ReshapeIsBitcast(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 2, 2}, {0, 1, 2}),
      ShapeUtil::MakeShapeWithDenseLayout(F32, {6, 2}, {0, 1})));
}

TEST(AlignmentTest, AlignLayoutsWithoutTrivialDimensions) {
  Shape input = ShapeUtil::MakeShapeWithDenseLayout(xla::F32, {3, 8, 5, 7, 11},
                                                    {3, 2, 1, 0, 4});
  auto aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {4, 3, 2, 7, 5, 11}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_THAT(aligned_shape.value().layout().minor_to_major(),
              ElementsAre(4, 3, 2, 1, 0, 5));
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(input, aligned_shape.value()));

  aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {3, 2, 4, 35, 11}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_THAT(aligned_shape.value().layout().minor_to_major(),
              ElementsAre(3, 2, 1, 0, 4));
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(input, aligned_shape.value()));
}

TEST(AlignmentTest, AlignLayoutsWithTrivialDimensions) {
  Shape input = ShapeUtil::MakeShapeWithDenseLayout(
      xla::F32, {1, 3, 8, 1, 5, 7, 1, 11, 1, 1},
      {5, 0, 4, 2, 1, 3, 6, 7, 9, 8});
  auto aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {1, 4, 1, 3, 2, 7, 5, 11, 1}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(input, aligned_shape.value()));
}

TEST(AlignmentTest, AlignLayoutsWithAllTrivialDimensions) {
  Shape input =
      ShapeUtil::MakeShapeWithDenseLayout(xla::F32, {1, 1, 1, 1}, {0, 1, 3, 2});
  auto aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {1, 1, 1, 1, 1}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(input, aligned_shape.value()));
}

// A test case where the consecutive elements of the input shape belonging to
// the same layout part are not in descending order.
TEST(AlignmentTest, AlignLayoutsWithoutTrivialDimensionsWrongInputLayout) {
  // Same physical layout as in AlignLayoutsWithoutTrivialDimensions, except
  // that the first two dimension numbers are exchanged.
  Shape input = ShapeUtil::MakeShapeWithDenseLayout(xla::F32, {3, 8, 5, 7, 11},
                                                    {2, 3, 1, 0, 4});
  auto aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {4, 3, 2, 7, 5, 11}));
  EXPECT_FALSE(aligned_shape);
}

// A test case where the physical layout of the input shape does not place all
// dimensions that belong to the same alignment part consecutively.
TEST(AlignmentTest,
     AlignLayoutsWithoutTrivialDimensionsNonConsecutiveAlignmentPart) {
  Shape input = ShapeUtil::MakeShapeWithDenseLayout(xla::F32, {3, 8, 5, 7, 11},
                                                    {3, 2, 1, 0, 4});
  auto aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {4, 3, 2, 5, 77}));
  EXPECT_FALSE(aligned_shape);
}

void BM_MakeShape(::testing::benchmark::State& state) {
  for (auto s : state) {
    ShapeUtil::MakeShape(F32, {2});
  }
}
BENCHMARK(BM_MakeShape);

void BM_MakeValidatedShape(::testing::benchmark::State& state) {
  for (auto s : state) {
    ShapeUtil::MakeValidatedShape(F32, {2}).value();
  }
}
BENCHMARK(BM_MakeValidatedShape);

Shape ShapeForBenchmark(::testing::benchmark::State& state) {
  Shape shape;
  switch (state.range(0)) {
    case 0: {
      shape = ShapeUtil::MakeShape(xla::F32, {1});
      break;
    }
    case 1: {
      shape = ShapeUtil::MakeShape(xla::F32, {4, 1});
      break;
    }
    case 2: {
      shape = ShapeUtil::MakeShape(xla::F32, {256, 1, 1024});
      break;
    }
  }
  state.SetLabel(shape.ToString());
  return shape;
}

void BM_ForEachIndex(::testing::benchmark::State& state) {
  Shape shape = ShapeForBenchmark(state);
  for (auto s : state) {
    int count = 0;
    auto increment_func =
        [&count](absl::Span<const int64_t> indexes) -> StatusOr<bool> {
      count++;
      return true;
    };
    ShapeUtil::ForEachIndex(shape, increment_func);
  }
}
BENCHMARK(BM_ForEachIndex)->Arg(0)->Arg(1)->Arg(2);

void BM_ForEachIndexNoStatus(::testing::benchmark::State& state) {
  Shape shape = ShapeForBenchmark(state);
  for (auto s : state) {
    int count = 0;
    auto increment_func = [&count](absl::Span<const int64_t> indexes) -> bool {
      count++;
      return true;
    };
    ShapeUtil::ForEachIndexNoStatus(shape, increment_func);
  }
}
BENCHMARK(BM_ForEachIndexNoStatus)->Arg(0)->Arg(1)->Arg(2);

}  // namespace
}  // namespace xla
