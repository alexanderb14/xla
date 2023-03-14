#define DATA_TYPE double

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/array.h"

#include <memory>
#include <string>

enum cmd_option {
  option_time,
  option_time_sequential,
  option_validate
};
cmd_option parseOption(int argc, char** argv);

std::shared_ptr<xla::PjRtStreamExecutorClient> buildJITClient(cmd_option option);
std::unique_ptr<xla::PjRtLoadedExecutable> buildExecutable(
    std::shared_ptr<xla::PjRtStreamExecutorClient> client, std::string program_path);

std::unique_ptr<xla::PjRtBuffer> buildBufferFromScalar(
    std::shared_ptr<xla::PjRtStreamExecutorClient> client,
    DATA_TYPE scalar);
std::unique_ptr<xla::PjRtBuffer> buildBuffer1D(
    std::shared_ptr<xla::PjRtStreamExecutorClient> client,
    xla::Array<DATA_TYPE>& arr);
std::unique_ptr<xla::PjRtBuffer> buildBuffer2D(
    std::shared_ptr<xla::PjRtStreamExecutorClient> client,
    xla::Array2D<DATA_TYPE>& arr);
std::unique_ptr<xla::PjRtBuffer> buildBuffer3D(
    std::shared_ptr<xla::PjRtStreamExecutorClient> client,
    xla::Array3D<DATA_TYPE>& arr);
