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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_

#include <memory>
#include <optional>
#include <vector>

#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct NcclAllReduceConfig {
  NcclCollectiveConfig config;
  ReductionKind reduction_kind;
};

// Thunk that performs a NCCL-based All-Reduce or Reduce-Scatter among CUDA
// GPU-based replicas.
class NcclAllReduceReduceScatterThunkBase : public NcclCollectiveThunk {
 public:
  static std::optional<ReductionKind> MatchAllReduceComputation(
      mlir::Region& computation);

  NcclAllReduceReduceScatterThunkBase(Kind kind, ThunkInfo thunk_info,
                                      NcclAllReduceConfig config,
                                      std::vector<Buffer> buffers);

 protected:
  const NcclCollectiveConfig& config() const override { return config_.config; }

  const NcclAllReduceConfig config_;
  const std::vector<Buffer> buffers_;
};

class NcclAllReduceThunkBase : public NcclAllReduceReduceScatterThunkBase {
 public:
  using NcclAllReduceReduceScatterThunkBase::
      NcclAllReduceReduceScatterThunkBase;

 protected:
  Status RunAllReduce(const ExecuteParams& params, se::Stream& stream,
                      ncclComm_t comm);
};

class NcclAllReduceThunk : public NcclAllReduceThunkBase {
 public:
  NcclAllReduceThunk(ThunkInfo thunk_info, mlir::lmhlo::AllReduceOp op,
                     std::vector<Buffer> buffers);

  static const char* GetName() { return "AllReduce"; }

  static bool CanImplement(mlir::lmhlo::AllReduceOp op);
  static bool IsDegenerate(mlir::lmhlo::AllReduceOp op, int64_t replica_count,
                           int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::AllReduceOp op);
  static constexpr bool IsAsync() { return false; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;
};

class NcclAllReduceStartThunk : public NcclAllReduceThunkBase {
 public:
  NcclAllReduceStartThunk(ThunkInfo thunk_info,
                          mlir::lmhlo_gpu::AllReduceStartOp op,
                          std::vector<Buffer> buffers);

  static const char* GetName() { return "AllReduceStart"; }

  static bool CanImplement(mlir::lmhlo_gpu::AllReduceStartOp op);
  static bool IsDegenerate(mlir::lmhlo_gpu::AllReduceStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::AllReduceStartOp op);
  static constexpr bool IsAsync() { return true; }

  AsyncExecutor& async_executor() { return async_; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;

 private:
  AsyncExecutor async_;
};

class NcclAllReduceDoneThunk : public NcclCollectiveDoneThunk {
 public:
  NcclAllReduceDoneThunk(ThunkInfo thunk_info,
                         NcclCollectiveThunk::AsyncExecutor& async);
};

class NcclReduceScatterThunk : public NcclAllReduceReduceScatterThunkBase {
 public:
  NcclReduceScatterThunk(ThunkInfo thunk_info, mlir::lmhlo::ReduceScatterOp op,
                         std::vector<Buffer> buffers);

  static const char* GetName() { return "ReduceScatter"; }

  // Returns whether the given instruction can be lowered to a nccl
  // reduce-scatter call.
  static bool CanImplement(mlir::lmhlo::ReduceScatterOp op);
  static bool IsDegenerate(mlir::lmhlo::ReduceScatterOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::ReduceScatterOp op);
  static constexpr bool IsAsync() { return false; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;
};

Status RunAllReduce(ReductionKind reduction_kind,
                    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm);

Status RunReduceScatter(ReductionKind reduction_kind,
                        std::vector<DeviceBufferPair>& buffers,
                        se::Stream& stream, ncclComm_t comm);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
