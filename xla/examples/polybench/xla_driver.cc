#define DATA_TYPE float

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/array.h"


enum cmd_option {
  option_time,
  option_time_sequential,
  option_validate
};

#include "xla_compiler.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/literal_util.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/platform_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"

#include <iostream>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>
#include <string>

using namespace xla;

cmd_option parseOption(int argc, char** argv) {
  if (argc < 2) {
    return option_time;
  }
  std::string option = argv[1];
  if (option == "--time") {
    return option_time;
  } else if (option == "--time-sequential") {
    return option_time_sequential;
  } else if (option == "--validate") {
    return option_validate;
  } else {
    std::cerr << "Usage: " << argv[0] << " --time | --time-sequential | --validate" << std::endl;
    exit(1);
  }
}

std::shared_ptr<PjRtStreamExecutorClient> buildJITClient(cmd_option option) {
  // Setup client
  //LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();

  LocalClientOptions options;
  if (option == option_time_sequential) {
    options.set_intra_op_parallelism_threads(1);
  }
  auto local_client_status = xla::ClientLibrary::GetOrCreateLocalClient(options);
  assert(local_client_status.ok());
  LocalClient* local_client = local_client_status.value();

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
  auto pjrt_se_client = std::make_shared<PjRtStreamExecutorClient>(
      "cpu", local_client, std::move(devices), /*process_index=*/0,
      /*allocator=*/nullptr, /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr);

  return pjrt_se_client;
}

std::unique_ptr<PjRtLoadedExecutable> buildExecutable(std::shared_ptr<PjRtStreamExecutorClient> client, std::string program_path) {
  // Read StableHLO program to string.
  //std::string program_path = tsl::io::JoinPath(
  //    tsl::testing::XlaSrcRoot(), "examples", "axpy", "stablehlo_axpy.mlir");
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
                          client->Compile(*program, CompileOptions{}).value();

  return executable;
}

std::unique_ptr<PjRtBuffer> buildBufferFromScalar(
    std::shared_ptr<PjRtStreamExecutorClient> client,
    DATA_TYPE scalar) {
  PjRtDevice* cpu = client->devices()[0];
  auto x_literal = xla::LiteralUtil::CreateR0<DATA_TYPE>(scalar);
  std::unique_ptr<PjRtBuffer> x =
      client->BufferFromHostLiteral(x_literal, cpu).value();
  auto sx = x->BlockHostUntilReady();

  return x;
}


std::unique_ptr<PjRtBuffer> buildBuffer1D(
    std::shared_ptr<PjRtStreamExecutorClient> client,
    xla::Array<DATA_TYPE>& arr) {
  PjRtDevice* cpu = client->devices()[0];
  auto x_literal = xla::LiteralUtil::CreateFromArray<DATA_TYPE>(arr);
  std::unique_ptr<PjRtBuffer> x =
      client->BufferFromHostLiteral(x_literal, cpu).value();
  auto sx = x->BlockHostUntilReady();

  return x;
}

std::unique_ptr<PjRtBuffer> buildBuffer2D(
    std::shared_ptr<PjRtStreamExecutorClient> client,
    xla::Array2D<DATA_TYPE>& arr) {
  PjRtDevice* cpu = client->devices()[0];
  auto x_literal = xla::LiteralUtil::CreateR2FromArray2D<DATA_TYPE>(arr);
  std::unique_ptr<PjRtBuffer> x =
      client->BufferFromHostLiteral(x_literal, cpu).value();
  auto sx = x->BlockHostUntilReady();

  return x;
}

std::unique_ptr<PjRtBuffer> buildBuffer3D(
    std::shared_ptr<PjRtStreamExecutorClient> client,
    xla::Array3D<DATA_TYPE>& arr) {
  PjRtDevice* cpu = client->devices()[0];
  auto x_literal = xla::LiteralUtil::CreateR3FromArray3D<DATA_TYPE>(arr);
  std::unique_ptr<PjRtBuffer> x =
      client->BufferFromHostLiteral(x_literal, cpu).value();
  auto sx = x->BlockHostUntilReady();

  return x;
}
