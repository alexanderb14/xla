/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_ALGORITHM_PICKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_ALGORITHM_PICKER_H_

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/time/time.h"
#include "xla/autotune_results.pb.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/gpu_serializable_autotuner.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/protobuf/autotuning.pb.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
#include "xla/stream_executor/gpu/redzone_allocator.h"
#endif

namespace xla {
namespace gpu {

// Modifies CustomCalls to cudnn convolutions, choosing the best algorithm for
// each and adding explicit scratch space to the CustomCalls.
//
// It supports two modes: device and deviceless.
// In device mode, we run autotuning on the device and store autotune results.
//
// In deviceless mode, we pass in some information related to the device and
// use stored autotune results to rewrite convolutions. If the required autotune
// result is not stored, then the performance of convolution will be suboptimal.
class GpuConvAlgorithmPicker : public HloModulePass {
 public:
  static void ClearAutotuneResults();
  static Status WriteAutotuneResults(AutotuneResults* results);
  static Status LoadAutotuneResults(const AutotuneResults& results);

  explicit GpuConvAlgorithmPicker(AutotuningConfig config) : config_(config) {}

  absl::string_view name() const override {
    return "gpu-conv-algorithm-picker";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation* computation);
  StatusOr<bool> RunOnInstruction(HloInstruction* instr);
  StatusOr<tensorflow::AutotuneResult> PickBestAlgorithm(
      const HloCustomCallInstruction* instr);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
  // Simple bundle of an algorithm and its output, for comparing results across
  // autotuned algorithms.
  struct ReferenceResult {
    stream_executor::dnn::AlgorithmDesc algorithm;
    stream_executor::DeviceMemoryBase buffer;
  };

  // Debug information about the instruction we are autotuning.
  struct AutotuneInstructionInfo {
    std::string instr_str;
    std::string module_str;

    explicit AutotuneInstructionInfo(const HloCustomCallInstruction* instr)
        : instr_str(instr->ToString()),
          module_str(instr->GetModule()->ToString()) {}
  };

  // Execution environment for autotuning. Runtime autotuning requires runtime
  // information such as input/output buffers in order to run. It can be
  // constructed from the autotuned instruction by FromInstruction.
  struct AutotuneRuntimeArguments {
    const Shape result_shape;
    const HloModuleConfig hlo_module_config;
    std::vector<se::DeviceMemoryBase> operand_buffers;
    se::DeviceMemoryBase result_buffer;
    se::RedzoneAllocator* input_output_allocator;
    const GpuConvConfig gpu_conv_config;
    std::string canonical_hlo;

    static StatusOr<AutotuneRuntimeArguments> FromInstruction(
        const HloCustomCallInstruction* instr,
        se::DeviceMemoryAllocator* allocator, se::StreamExecutor* stream,
        se::RedzoneAllocator* input_output_allocator);
  };

  StatusOr<tensorflow::AutotuneResult> AutotuneOneConvRunner(
      se::DeviceMemoryAllocator* allocator, se::Stream* stream,
      MaybeFusedConvRunner* const runner,
      std::optional<ReferenceResult>* reference_result,
      absl::Span<const stream_executor::dnn::AlgorithmDesc> disabled_algos,
      std::optional<AutotuneInstructionInfo> instruction_info,
      const AutotuneRuntimeArguments& runtime_arguments);

  // Pick the best algorithm for CUDA platform.
  StatusOr<tensorflow::AutotuneResult> PickBestAlgorithmNoCacheCuda(
      const HloCustomCallInstruction* instr,
      se::DeviceMemoryAllocator* allocator, se::Stream* stream,
      std::optional<AutotuneInstructionInfo> instruction_info,
      const AutotuneRuntimeArguments& runtime_arguments);
#endif

  StatusOr<tensorflow::AutotuneResult> PickBestAlgorithmNoCacheRocm(
      const HloCustomCallInstruction* instr,
      se::DeviceMemoryAllocator* allocator, se::Stream* stream);

 private:
  AutotuningConfig config_;
};

}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_ALGORITHM_PICKER_H_
