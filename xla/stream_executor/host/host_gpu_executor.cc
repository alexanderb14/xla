/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Implementation of HostExecutor class [of those methods not defined in the
// class declaration].
#include "xla/stream_executor/host/host_gpu_executor.h"

#include <stdint.h>
#include <string.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/host/host_stream.h"
#include "xla/stream_executor/host/host_timer.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/profile_utils/cpu_utils.h"

namespace stream_executor {
namespace host {

HostStream* AsHostStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<HostStream*>(stream->implementation());
}

HostExecutor::HostExecutor(const PluginConfig& plugin_config)
    : plugin_config_(plugin_config) {}

HostExecutor::~HostExecutor() {}

tsl::Status HostExecutor::Init(int device_ordinal,
                               DeviceOptions device_options) {
  auto it =
      device_options.non_portable_tags.find("host_thread_stack_size_in_bytes");
  if (it != device_options.non_portable_tags.end()) {
    if (!absl::SimpleAtoi(it->second, &thread_stack_size_in_bytes_)) {
      return tsl::errors::InvalidArgument(
          "Unable to parse host_thread_stack_size_in_bytes as an integer: ",
          it->second);
    }
  }
  return ::tsl::OkStatus();
}

bool HostExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  tsl::port::MemoryInfo mem_info = tsl::port::GetMemoryInfo();
  *free = (mem_info.free != INT64_MAX) ? mem_info.free : -1;
  *total = (mem_info.total != INT64_MAX) ? mem_info.total : -1;
  return true;
}

DeviceMemoryBase HostExecutor::Allocate(uint64_t size, int64_t memory_space) {
  CHECK_EQ(memory_space, 0);
  // Use a minimum alignment of 64 bytes to be friendly to AVX512 code.
  // This should probably be kept in sync with
  // tsl::Allocator::kAllocatorAlignment.
  return DeviceMemoryBase(
      tsl::port::AlignedMalloc(size, /*minimum_alignment=*/64), size);
}

void* HostExecutor::GetSubBuffer(DeviceMemoryBase* parent,
                                 uint64_t offset_bytes, uint64_t size_bytes) {
  return reinterpret_cast<char*>(parent->opaque()) + offset_bytes;
}

void HostExecutor::Deallocate(DeviceMemoryBase* mem) {
  tsl::port::AlignedFree(mem->opaque());
}

tsl::Status HostExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                             uint64_t size) {
  memset(location->opaque(), 0, size);
  return ::tsl::OkStatus();
}

tsl::Status HostExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                            int value, uint64_t size) {
  memset(location->opaque(), value, size);
  return ::tsl::OkStatus();
}

bool HostExecutor::Memcpy(Stream* stream, void* host_dst,
                          const DeviceMemoryBase& gpu_src, uint64_t size) {
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  AsHostStream(stream)->EnqueueTask(
      [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return true;
}

bool HostExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                          const void* host_src, uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return true;
}

bool HostExecutor::MemcpyDeviceToDevice(Stream* stream,
                                        DeviceMemoryBase* gpu_dst,
                                        const DeviceMemoryBase& gpu_src,
                                        uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  // Enqueue this [asynchronous] "device-to-device" (i.e., host-to-host, given
  // the nature of the HostExecutor) memcpy  on the stream (HostStream)
  // associated with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [src_mem, dst_mem, size]() { memcpy(dst_mem, src_mem, size); });
  return true;
}

tsl::Status HostExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                  uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size]() { memset(gpu_mem, 0, size); });
  return ::tsl::OkStatus();
}

tsl::Status HostExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                 uint8 pattern, uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return ::tsl::OkStatus();
}

tsl::Status HostExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                   uint32_t pattern, uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return ::tsl::OkStatus();
}

tsl::Status HostExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src,
                                            uint64_t size) {
  memcpy(gpu_dst->opaque(), host_src, size);
  return ::tsl::OkStatus();
}

tsl::Status HostExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64_t size) {
  memcpy(host_dst, gpu_src.opaque(), size);
  return ::tsl::OkStatus();
}

tsl::Status HostExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
  memcpy(gpu_dst->opaque(), gpu_src.opaque(), size);
  return ::tsl::OkStatus();
}

bool HostExecutor::HostCallback(Stream* stream,
                                absl::AnyInvocable<tsl::Status() &&> callback) {
  AsHostStream(stream)->EnqueueTaskWithStatus(std::move(callback));
  return true;
}

bool HostExecutor::AllocateStream(Stream* stream) { return true; }

void HostExecutor::DeallocateStream(Stream* stream) {}

bool HostExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  auto event = std::make_shared<absl::Notification>();
  AsHostStream(other)->EnqueueTask([event]() { event->Notify(); });
  AsHostStream(dependent)->EnqueueTask(
      [event]() { event->WaitForNotification(); });
  return true;
}

class HostEvent : public internal::EventInterface {
 public:
  HostEvent() : notification_(std::make_shared<absl::Notification>()) {}

  std::shared_ptr<absl::Notification>& notification() { return notification_; }

 private:
  // We use a std::shared_ptr here because the client may delete the HostEvent
  // object while there are still RecordEvent and WaitForEvent callbacks pending
  // on a stream.
  std::shared_ptr<absl::Notification> notification_;
};

std::unique_ptr<internal::EventInterface>
HostExecutor::CreateEventImplementation() {
  return std::unique_ptr<internal::EventInterface>(new HostEvent());
}

static HostEvent* AsHostEvent(Event* event) {
  DCHECK(event != nullptr);
  return static_cast<HostEvent*>(event->implementation());
}

tsl::Status HostExecutor::AllocateEvent(Event* /*event*/) {
  return ::tsl::OkStatus();
}

tsl::Status HostExecutor::DeallocateEvent(Event* /*event*/) {
  return ::tsl::OkStatus();
}

tsl::Status HostExecutor::RecordEvent(Stream* stream, Event* event) {
  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->notification();
  AsHostStream(stream)->EnqueueTask([notification]() {
    CHECK(!notification->HasBeenNotified());
    notification->Notify();
  });
  return ::tsl::OkStatus();
}

tsl::Status HostExecutor::WaitForEvent(Stream* stream, Event* event) {
  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->notification();
  AsHostStream(stream)->EnqueueTask(
      [notification]() { notification->WaitForNotification(); });
  return ::tsl::OkStatus();
}

Event::Status HostExecutor::PollForEventStatus(Event* event) {
  absl::Notification& notification = *AsHostEvent(event)->notification();
  return notification.HasBeenNotified() ? Event::Status::kComplete
                                        : Event::Status::kPending;
}

bool HostExecutor::StartTimer(Stream* stream, Timer* timer) {
  dynamic_cast<HostTimer*>(timer->implementation())->Start(stream);
  return true;
}

bool HostExecutor::StopTimer(Stream* stream, Timer* timer) {
  dynamic_cast<HostTimer*>(timer->implementation())->Stop(stream);
  return true;
}

tsl::Status HostExecutor::BlockHostUntilDone(Stream* stream) {
  return AsHostStream(stream)->BlockUntilDone();
}

tsl::StatusOr<std::unique_ptr<DeviceDescription>>
HostExecutor::CreateDeviceDescription(int device_ordinal) {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  // TODO(rspringer): How to report a value that's based in reality but that
  // doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
  builder.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);

  float cycle_counter_frequency = static_cast<float>(
      tsl::profile_utils::CpuUtils::GetCycleCounterFrequency());
  builder.set_clock_rate_ghz(cycle_counter_frequency / 1e9);

  builder.set_name("Host");
  builder.set_platform_version("Default Version");

  return builder.Build();
}

bool HostExecutor::SupportsBlas() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::BlasFactory>(kHostPlatformId,
                                                plugin_config_.blas())
      .ok();
}

blas::BlasSupport* HostExecutor::CreateBlas() {
  PluginRegistry* registry = PluginRegistry::Instance();
  tsl::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(kHostPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.value()(this);
}

bool HostExecutor::SupportsFft() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::FftFactory>(kHostPlatformId,
                                               plugin_config_.fft())
      .ok();
}

fft::FftSupport* HostExecutor::CreateFft() {
  PluginRegistry* registry = PluginRegistry::Instance();
  tsl::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(kHostPlatformId,
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.value()(this);
}

bool HostExecutor::SupportsRng() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::RngFactory>(kHostPlatformId,
                                               plugin_config_.rng())
      .ok();
}

rng::RngSupport* HostExecutor::CreateRng() {
  PluginRegistry* registry = PluginRegistry::Instance();
  tsl::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(kHostPlatformId,
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.value()(this);
}

std::unique_ptr<internal::StreamInterface>
HostExecutor::GetStreamImplementation() {
  return std::unique_ptr<internal::StreamInterface>(
      new HostStream(thread_stack_size_in_bytes_));
}

}  // namespace host
}  // namespace stream_executor
