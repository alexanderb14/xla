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

#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/literal_util.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/platform_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"

using namespace xla;

int main(int argc, char** argv) {
  // Setup client
  LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();

  // Retrieve the "platform" we intend to execute the computation on. The
  // concept of "platform" in XLA abstracts entirely everything needed to
  // interact with some hardware (compiler, runtime, etc.). New HW vendor
  // plugs into XLA by registering a new platform with a different string
  // key. For example for an Nvidia GPU change the following to:
  //   PlatformUtil::GetPlatform("CUDA"));
  se::Platform* platform = PlatformUtil::GetPlatform("cpu").value();
  se::StreamExecutorConfig config;
  config.ordinal = 0;
  se::StreamExecutor* executor = platform->GetExecutor(config).value();

  // LocalDeviceState and PjRtStreamExecutorDevice describes the state of a
  // device which can do computation or transfer buffers. This could represent a
  // GPU or accelerator, but we'll use the CPU for this example.
  auto device_state = std::make_unique<LocalDeviceState>(
      executor, local_client, LocalDeviceState::kSynchronous,
      /*max_inflight_computations=*/32,
      /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
  auto device = std::make_unique<PjRtStreamExecutorDevice>(
      0, std::move(device_state), "cpu");
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  devices.emplace_back(std::move(device));

  // The PjRtStreamExecutorClient will allow us to compile and execute
  // computations on the device we just configured.
  auto pjrt_se_client = PjRtStreamExecutorClient(
      "cpu", local_client, std::move(devices), /*process_index=*/0,
      /*allocator=*/nullptr, /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr);

  // Read StableHLO program to string.
  //std::string program_path = tsl::io::JoinPath(
  //    tsl::testing::XlaSrcRoot(), "examples", "axpy", "stablehlo_axpy.mlir");
  std::string program_path = "/tmp/xla_compile/synth_and_prep_fn.mlir";
  std::string program_string;

  auto readStatus = tsl::ReadFileToString(tsl::Env::Default(), program_path,
                                     &program_string);
  assert(readStatus.ok());

  std::cerr << "Loaded StableHLO program from " << program_path << ":\n"
            << program_string << std::endl;

  // Register MLIR dialects necessary to parse our program. In our case this is
  // just the Func dialect and StableHLO.
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::func::FuncDialect>();
  mlir::stablehlo::registerAllDialects(dialects);

  // Parse StableHLO program.
  auto ctx = std::make_unique<mlir::MLIRContext>(dialects);
  mlir::OwningOpRef<mlir::ModuleOp> program =
      mlir::parseSourceString<mlir::ModuleOp>(program_string, ctx.get());

  // Use our client to compile our StableHLO program to an executable.
  std::unique_ptr<PjRtLoadedExecutable> executable =
                          pjrt_se_client.Compile(*program, CompileOptions{}).value();

  // Create inputs to our computation.
  auto x_a = xla::Array3D<double>(250, 220, 270);
  for (int i = 0; i < x_a.dim(0); ++i) {
    for (int j = 0; j < x_a.dim(1); ++j) {
      for (int k = 0; k < x_a.dim(2); ++k) {
        x_a(i, j, k) = (double)(1) / i;
      }
    }
  }
  auto x_literal = xla::LiteralUtil::CreateR3FromArray3D<double>(
      x_a);

  auto y_a = xla::Array2D<double>(270, 270);
  for (int i = 0; i < y_a.dim(0); ++i) {
    for (int j = 0; j < y_a.dim(1); ++j) {
      y_a(i, j) = (double) 1;
    }
  }
  auto y_literal = xla::LiteralUtil::CreateR2FromArray2D<double>(
      y_a);

  // Get the host device.
  PjRtDevice* cpu = pjrt_se_client.devices()[0];

  // Transfer our literals to buffers. If we were using a GPU, these buffers
  // would correspond to device memory.
  std::unique_ptr<PjRtBuffer> x =
                          pjrt_se_client.BufferFromHostLiteral(x_literal, cpu).value();
  std::unique_ptr<PjRtBuffer> y =
                          pjrt_se_client.BufferFromHostLiteral(y_literal, cpu).value();

  // Block until the buffers are ready.
  auto sx = x->BlockHostUntilReady();
  auto sy = y->BlockHostUntilReady();

  // Time the execution of the computation.
  absl::Time start = absl::Now();

  // Do our computation.
  ::xla::ExecuteOptions options;
  options.execution_mode = ::xla::ExecuteOptions::ExecutionMode::kSynchronous;
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> axpy_result =
      executable->Execute({{x.get(), y.get()}}, options).value();

  auto buffer = axpy_result[0][0].get();
  auto status = buffer->BlockHostUntilReady();

  // Time the execution of the computation.
  absl::Time end = absl::Now();
  std::cerr << "Execution time: " << absl::ToDoubleSeconds(end - start) << "s\n";

  //// Convert result buffer back to literal.
  //TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> axpy_result_literal,
  //                        axpy_result[0][0]->ToLiteralSync());
  //std::cerr << "Computation output: " << *axpy_result_literal << std::endl;

}
