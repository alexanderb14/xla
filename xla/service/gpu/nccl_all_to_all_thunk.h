/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_TO_ALL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_TO_ALL_THUNK_H_

#include <vector>

#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct NcclAllToAllConfig {
  NcclCollectiveConfig config;
  bool has_split_dimension;
};

// Thunk that performs a NCCL-based All-to-All among CUDA GPU-based replicas.
class NcclAllToAllThunk : public NcclCollectiveThunk {
 public:
  NcclAllToAllThunk(ThunkInfo thunk_info, mlir::lmhlo::AllToAllOp op,
                    std::vector<Buffer> buffers);

  // Returns whether the given instruction can be lowered to a nccl all-to-all
  // call.
  static bool CanImplement(mlir::lmhlo::AllToAllOp op);

  static const char* GetName() { return "AllToAll"; }
  static bool IsDegenerate(mlir::lmhlo::AllToAllOp op, int64_t replica_count,
                           int64_t partition_count) {
    return GetNcclAllToAllConfig(op).config.IsDegenerate(replica_count,
                                                         partition_count);
  }

  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::AllToAllOp op) {
    return GetNcclAllToAllConfig(op).config.group_mode;
  }
  static constexpr bool IsAsync() { return false; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;

  const NcclCollectiveConfig& config() const override { return config_.config; }

 private:
  static NcclAllToAllConfig GetNcclAllToAllConfig(mlir::lmhlo::AllToAllOp op);

  const NcclAllToAllConfig config_;
  const std::vector<Buffer> buffers_;
};

Status RunAllToAll(bool has_split_dimension,
                   std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                   ncclComm_t comm);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_TO_ALL_THUNK_H_
