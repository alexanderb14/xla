#define DATA_TYPE float

#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/Parser/Parser.h"            // from @llvm-project
#include "stablehlo/dialect/Register.h"    // from @stablehlo
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "xla/array.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/literal_util.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/platform_util.h"

using namespace xla;

std::unique_ptr<PjRtBuffer> buildBufferFromScalar(
    std::shared_ptr<PjRtStreamExecutorClient> client, DATA_TYPE scalar) {
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

std::unique_ptr<PjRtBuffer> buildBuffer4D(
    std::shared_ptr<PjRtStreamExecutorClient> client,
    xla::Array4D<DATA_TYPE>& arr) {
  PjRtDevice* cpu = client->devices()[0];
  auto x_literal = xla::LiteralUtil::CreateR4FromArray4D<DATA_TYPE>(arr);
  std::unique_ptr<PjRtBuffer> x =
      client->BufferFromHostLiteral(x_literal, cpu).value();
  auto sx = x->BlockHostUntilReady();

  return x;
}

std::shared_ptr<PjRtStreamExecutorClient> buildJITClient() {
  // Setup client
  // LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();

  LocalClientOptions options;
  // if (option == option_time_sequential) {
  //   options.set_intra_op_parallelism_threads(1);
  // }
  auto local_client_status =
      xla::ClientLibrary::GetOrCreateLocalClient(options);
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

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <mlir file>\n";
    return 1;
  }
  std::string program_file = argv[1];

  std::string program_string;

  if (program_file == "-") {
    // Read program from stdin.
    std::string line;
    while (std::getline(std::cin, line)) {
      program_string += line + "\n";
    }
  } else {
    // Read program from file.
    auto readStatus =
        tsl::ReadFileToString(tsl::Env::Default(), program_file, &program_string);
    assert(readStatus.ok());
    std::cerr << "Loaded program from " << program_file << ":\n"
              << program_string << std::endl;
  }

  auto client = buildJITClient();

  // Register MLIR dialects necessary to parse our program. In our case this is
  // just the Func dialect and StableHLO.
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::func::FuncDialect>();
  mlir::stablehlo::registerAllDialects(dialects);

  // Parse StableHLO program.
  auto ctx = std::make_unique<mlir::MLIRContext>(dialects);
  mlir::OwningOpRef<mlir::ModuleOp> program =
      mlir::parseSourceString<mlir::ModuleOp>(program_string, ctx.get());

  // Build args
  auto function = program->lookupSymbol<mlir::func::FuncOp>("main");
  std::vector<std::unique_ptr<PjRtBuffer>> args;
  for (auto arg : function.getArguments()) {
    std::unique_ptr<PjRtBuffer> buffer;

    auto shape = arg.getType().cast<mlir::ShapedType>().getShape();
    if (shape.size() == 0) {
      buffer = buildBufferFromScalar(client, 1.234);
    } else if (shape.size() == 1) {
      auto arr = xla::Array<DATA_TYPE>({shape[0]});
      buffer = buildBuffer1D(client, arr);
    } else if (shape.size() == 2) {
      auto arr = xla::Array2D<DATA_TYPE>(shape[0], shape[1]);
      buffer = buildBuffer2D(client, arr);
    } else if (shape.size() == 3) {
      auto arr = xla::Array3D<DATA_TYPE>(shape[0], shape[1], shape[2]);
      buffer = buildBuffer3D(client, arr);
    } else if (shape.size() == 4) {
      auto arr = xla::Array4D<DATA_TYPE>(shape[0], shape[1], shape[2], shape[3]);
      buffer = buildBuffer4D(client, arr);
    } else {
      std::cerr << "Unsupported shape size: " << shape.size() << std::endl;
      exit(1);
    }
    args.push_back(std::move(buffer));
  }

  // Convert args to the right type
  std::vector<PjRtBuffer*> args_ptrs;
  for (auto& arg : args) {
    args_ptrs.push_back(arg.get());
  }
  absl::Span<const std::vector<PjRtBuffer*>> args_span(&args_ptrs, 1);

  // Use our client to compile our StableHLO program to an executable.
  std::unique_ptr<PjRtLoadedExecutable> executable =
      client->Compile(*program, CompileOptions{}).value();

  // Run the executable.
  auto start_time = std::chrono::steady_clock::now();

  ::xla::ExecuteOptions options;
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result =
      executable->Execute(args_span, options).value();

  auto buffer = result[0][0].get();
  auto status = buffer->BlockHostUntilReady();

  auto end_time = std::chrono::steady_clock::now();
  std::cout << std::fixed << std::setprecision(7)
            << std::chrono::duration<double>(end_time - start_time).count()
            << "s\n";

  return 0;
}
