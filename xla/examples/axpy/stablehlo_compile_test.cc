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

#include "polybench.h"
#include "doitgen.h"

using namespace xla;

/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (DATA_TYPE) (i*j % np) / np;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgen(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

#pragma scop
  for (r = 0; r < _PB_NR; r++)
    for (q = 0; q < _PB_NQ; q++)  {
      for (p = 0; p < _PB_NP; p++)  {
	sum[p] = SCALAR_VAL(0.0);
	for (s = 0; s < _PB_NP; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < _PB_NP; p++)
	A[r][q][p] = sum[p];
    }
#pragma endscop

}

std::shared_ptr<PjRtStreamExecutorClient> buildJITClient() {
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



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum,DATA_TYPE,NP,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

  // Build executable
  auto client = buildJITClient();
  auto executable = buildExecutable(client, "/tmp/xla_compile/synth_and_prep_fn.mlir");

  // Create inputs to our computation.
  PjRtDevice* cpu = client->devices()[0];

  auto x_a = xla::Array3D<double>(NR, NQ, NP);
  for (int i = 0; i < x_a.dim(0); ++i) {
    for (int j = 0; j < x_a.dim(1); ++j) {
      for (int k = 0; k < x_a.dim(2); ++k) {
	      x_a(i, j, k) = (*A)[i][j][k];
      }
    }
  }
  auto x_literal = xla::LiteralUtil::CreateR3FromArray3D<double>(
      x_a);
  std::unique_ptr<PjRtBuffer> x =
                          client->BufferFromHostLiteral(x_literal, cpu).value();

  auto y_a = xla::Array2D<double>(NP, NP);
  for (int i = 0; i < y_a.dim(0); ++i) {
    for (int j = 0; j < y_a.dim(1); ++j) {
      y_a(i, j) = (*C4)[i][j];
    }
  }
  auto y_literal = xla::LiteralUtil::CreateR2FromArray2D<double>(
      y_a);
  std::unique_ptr<PjRtBuffer> y =
                          client->BufferFromHostLiteral(y_literal, cpu).value();

  // Block until the buffers are ready.
  auto sx = x->BlockHostUntilReady();
  auto sy = y->BlockHostUntilReady();

  /* Start timer. */
  polybench_start_instruments;

  ::xla::ExecuteOptions options;
  options.execution_mode = ::xla::ExecuteOptions::ExecutionMode::kSynchronous;

  /* Run kernel. */
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> axpy_result =
      executable->Execute({{x.get(), y.get()}}, options).value();

  auto buffer = axpy_result[0][0].get();
  auto status = buffer->BlockHostUntilReady();

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  // Store the data in the result literal back in the array A.
  std::shared_ptr<Literal> axpy_result_literal = 
                          axpy_result[0][0]->ToLiteralSync().value();
  auto axpy_result_a = axpy_result_literal->data<double>();
  for (int i = 0; i < x_a.dim(0); ++i) {
    for (int j = 0; j < x_a.dim(1); ++j) {
      for (int k = 0; k < x_a.dim(2); ++k) {
        (*A)[i][j][k] = axpy_result_a[i*x_a.dim(1)*x_a.dim(2) + j*x_a.dim(2) + k];
      }
    }
  }

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  return 0;
}

