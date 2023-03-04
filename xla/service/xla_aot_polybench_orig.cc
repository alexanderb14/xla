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

#include <string>

#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"

using namespace xla;

xla::Status RunMain() {
  std::string path = "/tmp/xla_compile/compiled.xla";
  std::string serialized_aot_result;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), path, &serialized_aot_result));

  // Get a LocalClient
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                          PlatformUtil::GetPlatform("Host"));
  if (platform->VisibleDeviceCount() <= 0) {
    assert(falst && "CPU platform has no visible devices.");
  }
  LocalClientOptions local_client_options;
  local_client_options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(
      LocalClient * client,
      ClientLibrary::GetOrCreateLocalClient(local_client_options));

  // Load from AOT result.
  ExecutableBuildOptions executable_build_options;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LocalExecutable> local_executable,
      client->Load(serialized_aot_result, executable_build_options));

  // Run loaded excutable.
  auto x_literal = xla::LiteralUtil::CreateR1<double>({1.0f, 2.0f, 3.0f});
  auto y_literal = xla::LiteralUtil::CreateR1<double>({1.0f, 2.0f, 3.0f});
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer x,
                          client->LiteralToShapedBuffer(
                              x_literal, client->default_device_ordinal()));
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer y,
                          client->LiteralToShapedBuffer(
                              y_literal, client->default_device_ordinal()));
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_allocator(client->backend().memory_allocator());
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer axpy_result,
      local_executable->Run((absl::Span<const ShapedBuffer* const>) {&x, &y}, executable_run_options));

  TF_ASSIGN_OR_RETURN(Literal axpy_result_literal,
                          client->ShapedBufferToLiteral(axpy_result));

  std::cout << "axpy_result_literal: " << axpy_result_literal.ToString();

  //TF_ASSIGN_OR_RETURN(Literal output,
  //                        client->ShapedBufferToLiteral(result));

  return tsl::OkStatus();
}

int main(int argc, char* argv[]) {
  xla::Status result = RunMain();
  if (!result.ok()) {
    LOG(ERROR) << result;
    return 1;
  }
  return 0;
}
