/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/cpu/cpu_runtime.h"

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/layout_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/cpu/cpu_executable_run_options.h"
#include "xla/service/cpu/in_process_collectives.h"
#include "xla/service/cpu/xfeed_manager.h"
#include "xla/service/global_device_id.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace cpu {
namespace runtime {

XfeedManager* GetXfeedManager(int device_ordinal) {
  static auto* managers = new absl::flat_hash_map<int, XfeedManager*>();
  static absl::Mutex* mutex = new absl::Mutex();

  absl::MutexLock lock(mutex);
  auto it = managers->find(device_ordinal);
  if (it == managers->end()) {
    it = managers->emplace(device_ordinal, new XfeedManager()).first;
  }
  return it->second;
}

extern const char* const kEigenMatMulF16SymbolName =
    "__xla_cpu_runtime_EigenMatMulF16";
extern const char* const kEigenMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenMatMulF32";
extern const char* const kEigenMatMulF64SymbolName =
    "__xla_cpu_runtime_EigenMatMulF64";
extern const char* const kEigenMatMulC64SymbolName =
    "__xla_cpu_runtime_EigenMatMulC64";
extern const char* const kEigenMatMulC128SymbolName =
    "__xla_cpu_runtime_EigenMatMulC128";
extern const char* const kEigenMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenMatMulS32";
extern const char* const kEigenBatchMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenBatchMatMulF32";
extern const char* const kMKLConv2DF32SymbolName =
    "__xla_cpu_runtime_MKLConv2DF32";
extern const char* const kACLConv2DF32SymbolName =
    "__xla_cpu_runtime_ACLConv2DF32";
extern const char* const kACLMatMulF32SymbolName =
    "__xla_cpu_runtime_ACLMatMulF32";
extern const char* const kACLBatchMatMulF32SymbolName =
    "__xla_cpu_runtime_ACLBatchMatMulF32";
extern const char* const kEigenConv2DF16SymbolName =
    "__xla_cpu_runtime_EigenConv2DF16";
extern const char* const kEigenConv2DF32SymbolName =
    "__xla_cpu_runtime_EigenConv2DF32";
extern const char* const kEigenConv3DF16SymbolName =
    "__xla_cpu_runtime_EigenConv3DF16";
extern const char* const kEigenConv3DF32SymbolName =
    "__xla_cpu_runtime_EigenConv3DF32";
extern const char* const kDuccFftSymbolName = "__xla_cpu_runtime_DuccFft";
extern const char* const kDuccSingleThreadedFftSymbolName =
    "__xla_cpu_runtime_DuccSingleThreadedFft";
extern const char* const kEigenSingleThreadedMatMulF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF16";
extern const char* const kEigenSingleThreadedMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF32";
extern const char* const kEigenSingleThreadedMatMulF64SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF64";
extern const char* const kEigenSingleThreadedMatMulC64SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulC64";
extern const char* const kEigenSingleThreadedMatMulC128SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulC128";
extern const char* const kEigenSingleThreadedMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulS32";
extern const char* const kEigenSingleThreadedConv2DF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv2DF16";
extern const char* const kEigenSingleThreadedConv2DF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv2DF32";
extern const char* const kEigenSingleThreadedConv3DF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv3DF16";
extern const char* const kEigenSingleThreadedConv3DF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv3DF32";
extern const char* const kAcquireInfeedBufferForDequeueSymbolName =
    "__xla_cpu_runtime_AcquireInfeedBufferForDequeue";
extern const char* const kReleaseInfeedBufferAfterDequeueSymbolName =
    "__xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue";
extern const char* const kAcquireOutfeedBufferForPopulationSymbolName =
    "__xla_cpu_runtime_AcquireOutfeedBufferForPopulation";
extern const char* const kReleaseOutfeedBufferAfterPopulationSymbolName =
    "__xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation";
extern const char* const kParallelForkJoinSymbolName =
    "__xla_cpu_runtime_ParallelForkJoin";
extern const char* const kPrintfToStderrSymbolName =
    "__xla_cpu_runtime_PrintfToStderr";
extern const char* const kStatusIsSuccessSymbolName =
    "__xla_cpu_runtime_StatusIsSuccess";
extern const char* const kKeyValueSortSymbolName =
    "__xla_cpu_runtime_KeyValueSort";
extern const char* const kTopKF32SymbolName = "__xla_cpu_runtime_TopKF32";
extern const char* const kTracingStartSymbolName =
    "__xla_cpu_runtime_TracingStart";
extern const char* const kTracingEndSymbolName = "__xla_cpu_runtime_TracingEnd";
extern const char* const kXlaCpuRuntimeSymbolNamePrefix = "__xla_cpu_runtime_";
extern const char* const kAllReduceSymbolName = "__xla_cpu_runtime_AllReduce";

extern const char* const kAllReduceStartSymbolName = "__xla_cpu_runtime_AllReduceStart";
extern const char* const kAllReduceDoneSymbolName = "__xla_cpu_runtime_AllReduceDone";
extern const char* const kAsyncDoneSymbolName = "__xla_cpu_runtime_AsyncDone";


extern const char* const kAllGatherSymbolName = "__xla_cpu_runtime_AllGather";
extern const char* const kReduceScatterSymbolName =
    "__xla_cpu_runtime_ReduceScatter";
extern const char* const kAllToAllSymbolName = "__xla_cpu_runtime_AllToAll";
extern const char* const kCollectivePermuteSymbolName =
    "__xla_cpu_runtime_CollectivePermute";
extern const char* const kPartitionIdSymbolName =
    "__xla_cpu_runtime_PartitionId";
extern const char* const kReplicaIdSymbolName = "__xla_cpu_runtime_ReplicaId";
extern const char* const kOneDnnMatMulSymbolName =
    "__xla_cpu_runtime_OneDnnMatMul";
extern const char* const kOneDnnSoftmaxSymbolName =
    "__xla_cpu_runtime_OneDnnSoftmax";
extern const char* const kOneDnnLayerNormSymbolName =
    "__xla_cpu_runtime_OneDnnLayerNorm";

namespace {

// Inverses the encoding of a Shape protobuf into an LLVM global variable.
StatusOr<Shape> DecodeSelfDescribingShapeConstant(const void* shape_ptr,
                                                  int32_t size_bytes) {
  ShapeProto shape_proto;
  if (!shape_proto.ParseFromArray(shape_ptr, size_bytes)) {
    return tsl::errors::Internal("Failed parsing the shape proto");
  }
  Shape shape(shape_proto);
  auto status = ShapeUtil::ValidateShape(shape);
  if (!status.ok()) {
    return status;
  }
  return std::move(shape);
}

std::string ShapeString(const void* shape_ptr, int32_t shape_length) {
  StatusOr<Shape> shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length);
  if (shape.ok()) {
    return ShapeUtil::HumanStringWithLayout(shape.value());
  }
  return "<invalid shape>";
}

// TODO(zhangqiaorjc): Prefer to make callers set and use device_ordinal
// directly since callers may not have a Stream*.
int GetDeviceOrdinal(const ExecutableRunOptions* run_options) {
  if (!run_options) {
    return 0;
  } else if (run_options->device_ordinal() != -1) {
    return run_options->device_ordinal();
  }
  return run_options->stream()->parent()->device_ordinal();
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void* AcquireInfeedBufferForDequeueImpl(const ExecutableRunOptions* run_options,
                                        int32_t buffer_length,
                                        const void* shape,
                                        int32_t shape_length) {
  int device_ordinal = GetDeviceOrdinal(run_options);

  VLOG(2) << "AcquireInfeedBufferForDequeue: "
          << ShapeString(shape, shape_length) << " on stream executor "
          << device_ordinal;

  XfeedManager* xfeed = GetXfeedManager(device_ordinal);
  // Wait until there's a buffer to dequeue.
  XfeedBuffer* buffer = xfeed->infeed()->BlockingDequeueBuffer();
  CHECK_EQ(buffer->length(), buffer_length)
      << "XLA program infeed request buffer size " << buffer_length
      << " did not match the runtime's infed buffer length " << buffer->length()
      << "; program reports desired shape: "
      << ShapeString(shape, shape_length);
  return buffer->data();
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void ReleaseInfeedBufferAfterDequeueImpl(
    const ExecutableRunOptions* run_options, int32_t buffer_length,
    void* buffer_ptr, const void* shape_ptr, int32_t shape_length) {
  int device_ordinal = GetDeviceOrdinal(run_options);

  VLOG(2) << "ReleaseInfeedBufferAfterDeque: "
          << ShapeString(shape_ptr, shape_length) << " on stream executor "
          << device_ordinal;

  XfeedManager* xfeed = GetXfeedManager(device_ordinal);
  StatusOr<Shape> shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length);
  xfeed->infeed()->ReleaseCurrentBuffer(buffer_length, buffer_ptr,
                                        std::move(shape));
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void* AcquireOutfeedBufferForPopulationImpl(
    const ExecutableRunOptions* run_options, int32_t buffer_length,
    const void* shape_ptr, int32_t shape_length) {
  int device_ordinal = GetDeviceOrdinal(run_options);

  VLOG(2) << "AcquireOutfeedBufferForPopulation: "
          << ShapeString(shape_ptr, shape_length) << " on stream executor "
          << device_ordinal;

  XfeedManager* xfeed = GetXfeedManager(device_ordinal);
  // Wait until there's a buffer to dequeue.
  XfeedBuffer* buffer = xfeed->outfeed()->BlockingDequeueBuffer();
  CHECK_EQ(buffer->length(), buffer_length)
      << "XLA program outfeed request buffer size " << buffer_length
      << " did not match the runtime's outfeed buffer length "
      << buffer->length() << "; program reports outfed shape: "
      << ShapeString(shape_ptr, shape_length);
  return buffer->data();
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void ReleaseOutfeedBufferAfterPopulationImpl(
    const ExecutableRunOptions* run_options, int32_t buffer_length,
    void* buffer_ptr, const void* shape_ptr, int32_t shape_length) {
  int device_ordinal = GetDeviceOrdinal(run_options);

  VLOG(2) << "ReleaseOutfeedBufferAfterPopulation: "
          << ShapeString(shape_ptr, shape_length) << " on stream executor "
          << device_ordinal;

  XfeedManager* xfeed = GetXfeedManager(device_ordinal);
  StatusOr<Shape> shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length);
  xfeed->outfeed()->ReleaseCurrentBuffer(buffer_length, buffer_ptr,
                                         std::move(shape));
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void ReplicaIdImpl(const ExecutableRunOptions* run_options,
                   void* output_buffer) {
  int device_ordinal = GetDeviceOrdinal(run_options);
  int32_t replica_id = run_options->device_assignment()
                           ->ReplicaIdForDevice(GlobalDeviceId(device_ordinal))
                           .value();
  std::memcpy(output_buffer, &replica_id, 4);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void PartitionIdImpl(const ExecutableRunOptions* run_options,
                     void* output_buffer) {
  int device_ordinal = GetDeviceOrdinal(run_options);
  const DeviceAssignment::LogicalID logical_id =
      run_options->device_assignment()
          ->LogicalIdForDevice(GlobalDeviceId(device_ordinal))
          .value();
  std::memcpy(output_buffer, &logical_id.computation_id, 4);
}

RendezvousKey GetRendezvousKey(const ExecutableRunOptions* run_options,
                               GlobalDeviceId device,
                               std::vector<ReplicaGroup> group,
                               int32_t channel_id_present,
                               std::optional<bool> use_global_device_ids,
                               int64_t op_id) {
  const DeviceAssignment& device_assignment = *run_options->device_assignment();
  RendezvousKey::CollectiveOpKind op_kind = channel_id_present
                                                ? RendezvousKey::kCrossModule
                                                : RendezvousKey::kCrossReplica;
  std::vector<GlobalDeviceId> participating_devices =
      GetParticipatingDevices(GlobalDeviceId(device), device_assignment, group,
                              GetCollectiveOpGroupMode(channel_id_present != 0,
                                                       use_global_device_ids)
                                  .value())
          .value();
  int num_local_participants = participating_devices.size();
  return RendezvousKey{run_options->run_id(), std::move(participating_devices),
                       num_local_participants, op_kind, op_id};
}

CollectivesInterface* GetInProcessCollectivesImpl() {
  static InProcessCollectives* c = new InProcessCollectives();
  return c;
}

CollectivesInterface* GetCollectivesImpl(
    const ExecutableRunOptions* run_options) {
  if (run_options->cpu_executable_run_options() &&
      run_options->cpu_executable_run_options()->collectives()) {
    return run_options->cpu_executable_run_options()->collectives();
  }
  return GetInProcessCollectivesImpl();
}

absl::Duration DefaultCollectiveTimeout() { return absl::Minutes(30); }

absl::StatusOr<int> RankInGlobalDevices(
    absl::Span<GlobalDeviceId const> devices, GlobalDeviceId device) {
  auto it = absl::c_find(devices, device);
  if (it == devices.end()) {
    return InvalidArgument(
        "Device %d not present in global devices %s.", device.value(),
        absl::StrJoin(devices, ", ", [](std::string* out, GlobalDeviceId id) {
          absl::StrAppend(out, id.value());
        }));
  }
  return std::distance(devices.begin(), it);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void AllToAllImpl(const ExecutableRunOptions* run_options,
                  int32_t channel_id_present, int64_t op_id,
                  const void* replica_groups_str,
                  int32_t replica_groups_str_size, int32_t num_buffers,
                  int64_t buffer_size, void** source_buffers,
                  void** destination_buffers) {
  GlobalDeviceId device(GetDeviceOrdinal(run_options));
  std::string_view replica_groups_serialized(
      static_cast<const char*>(replica_groups_str), replica_groups_str_size);
  std::vector<ReplicaGroup> group =
      ParseReplicaGroupsOnly(replica_groups_serialized).value();
  RendezvousKey rendezvous_key =
      GetRendezvousKey(run_options, device, group, channel_id_present,
                       /*use_global_device_ids=*/std::nullopt, op_id);

  int rank = RankInGlobalDevices(rendezvous_key.global_devices, device).value();

  CollectivesInterface* collectives = GetCollectivesImpl(run_options);

  auto communicator =
      collectives->GetCommunicator(rendezvous_key.global_devices, rank).value();
  TF_CHECK_OK(communicator->AllToAll(
      rendezvous_key, buffer_size,
      absl::Span<const void* const>(source_buffers, num_buffers),
      absl::Span<void* const>(destination_buffers, num_buffers),
      DefaultCollectiveTimeout()));
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void AllGatherImpl(const ExecutableRunOptions* run_options,
                   int32_t channel_id_present, int32_t use_global_device_ids,
                   int64_t op_id, const void* replica_groups_str,
                   int32_t replica_groups_str_size, int64_t buffer_size,
                   void* source_buffer, void* destination_buffer) {
  GlobalDeviceId device(GetDeviceOrdinal(run_options));
  std::string_view replica_groups_serialized(
      static_cast<const char*>(replica_groups_str), replica_groups_str_size);
  std::vector<ReplicaGroup> group =
      ParseReplicaGroupsOnly(replica_groups_serialized).value();
  RendezvousKey rendezvous_key =
      GetRendezvousKey(run_options, device, group, channel_id_present,
                       use_global_device_ids, op_id);

  int rank = RankInGlobalDevices(rendezvous_key.global_devices, device).value();

  CollectivesInterface* collectives = GetCollectivesImpl(run_options);

  auto communicator =
      collectives->GetCommunicator(rendezvous_key.global_devices, rank).value();
  TF_CHECK_OK(communicator->AllGather(rendezvous_key, buffer_size,
                                      source_buffer, destination_buffer,
                                      DefaultCollectiveTimeout()));
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void ReduceScatterImpl(const ExecutableRunOptions* run_options,
                       const void* replica_groups_str,
                       int32_t replica_groups_str_size,
                       int32_t channel_id_present,
                       int32_t use_global_device_ids, int64_t op_id,
                       int32_t reduction_kind, int32_t element_type,
                       int64_t chunk_elems, void* input_buffer,
                       void* output_buffer) {
  GlobalDeviceId device(GetDeviceOrdinal(run_options));
  std::string_view replica_groups_serialized(
      static_cast<const char*>(replica_groups_str), replica_groups_str_size);
  std::vector<ReplicaGroup> group =
      ParseReplicaGroupsOnly(replica_groups_serialized).value();
  RendezvousKey rendezvous_key =
      GetRendezvousKey(run_options, device, group, channel_id_present,
                       use_global_device_ids, op_id);

  int rank = RankInGlobalDevices(rendezvous_key.global_devices, device).value();

  CollectivesInterface* collectives = GetCollectivesImpl(run_options);

  auto communicator =
      collectives->GetCommunicator(rendezvous_key.global_devices, rank).value();
  TF_CHECK_OK(communicator->ReduceScatter(
      rendezvous_key, static_cast<ReductionKind>(reduction_kind),
      static_cast<PrimitiveType>(element_type), chunk_elems, input_buffer,
      output_buffer, DefaultCollectiveTimeout()));
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void AllReduceImpl(const ExecutableRunOptions* run_options,
                   const void* replica_groups_str,
                   int32_t replica_groups_str_size, int32_t channel_id_present,
                   int32_t use_global_device_ids, int64_t op_id,
                   int32_t reduction_kind, const void* shape_ptr,
                   int32_t shape_length, int32_t num_buffers,
                   void** input_buffers, void** output_buffers, bool is_async) {
  std::cout << "AllReduceImpl (cpu_runtime.cc) is_async=" << is_async << std::endl;
  GlobalDeviceId device(GetDeviceOrdinal(run_options));
  std::string_view replica_groups_serialized(
      static_cast<const char*>(replica_groups_str), replica_groups_str_size);
  std::vector<ReplicaGroup> group =
      ParseReplicaGroupsOnly(replica_groups_serialized).value();
  RendezvousKey rendezvous_key =
      GetRendezvousKey(run_options, device, group, channel_id_present,
                       use_global_device_ids, op_id);
  auto shape_str = ShapeString(shape_ptr, shape_length);
  std::cout << "All-reduce input/output shape : " << shape_str << std::endl;

  Shape shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length).value();

  CHECK((num_buffers > 1 && shape.IsTuple()) ||
        (num_buffers == 1 && LayoutUtil::IsDenseArray(shape)));

  int rank = RankInGlobalDevices(rendezvous_key.global_devices, device).value();

  CollectivesInterface* collectives = GetCollectivesImpl(run_options);

  auto communicator =
      collectives->GetCommunicator(rendezvous_key.global_devices, rank).value();
  std::cout << "UU num_buffers " << num_buffers << std::endl;
  for (int i = 0; i < num_buffers; i++) {
    Shape subshape = num_buffers == 1 ? shape : shape.tuple_shapes(i);

    // if (is_async){
    //   subshape = subshape.tuple_shapes(1);
    // }

    // TODO!!! actually call async version
    TF_CHECK_OK(communicator->AllReduce(
        rendezvous_key, static_cast<ReductionKind>(reduction_kind),
        subshape.element_type(), ShapeUtil::ElementsIn(subshape),
        input_buffers[i], output_buffers[i], DefaultCollectiveTimeout(), is_async));
  }
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void DoneImpl(const ExecutableRunOptions* run_options,
                   const void* replica_groups_str,
                   int32_t replica_groups_str_size, int32_t channel_id_present,
                   int32_t use_global_device_ids, int64_t op_id) {
  std::cout << "DoneImpl (cpu_runtime.cc)=" << std::endl;
  GlobalDeviceId device(GetDeviceOrdinal(run_options));
  std::string_view replica_groups_serialized(
      static_cast<const char*>(replica_groups_str), replica_groups_str_size);
  std::vector<ReplicaGroup> group =
      ParseReplicaGroupsOnly(replica_groups_serialized).value();
  RendezvousKey rendezvous_key =
      GetRendezvousKey(run_options, device, group, channel_id_present,
                       use_global_device_ids, op_id);
  int rank = RankInGlobalDevices(rendezvous_key.global_devices, device).value();

  CollectivesInterface* collectives = GetCollectivesImpl(run_options);

  auto communicator =
      collectives->GetCommunicator(rendezvous_key.global_devices, rank).value();

  TF_CHECK_OK(communicator->WaitAll(rendezvous_key));
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
void CollectivePermuteImpl(const ExecutableRunOptions* run_options,
                           int32_t channel_id_present, int64_t op_id,
                           int32_t byte_size, void* input_buffer,
                           void* output_buffer, const void* source_target_pairs,
                           int32_t source_target_pairs_size) {
  GlobalDeviceId device(GetDeviceOrdinal(run_options));
  std::string_view source_target_pairs_serialized(
      static_cast<const char*>(source_target_pairs), source_target_pairs_size);
  auto pairs = absl::StrSplit(source_target_pairs_serialized, ',');
  const DeviceAssignment::LogicalID logical_id =
      run_options->device_assignment()->LogicalIdForDevice(device).value();
  int32_t logical_device_id =
      channel_id_present ? logical_id.computation_id : logical_id.replica_id;

  std::optional<int> source_replica_id;
  std::vector<int> copy_to;
  for (auto& p : pairs) {
    std::vector<std::string> mapping = absl::StrSplit(p, '=');
    CHECK_EQ(mapping.size(), 2);
    int from = std::stoi(mapping[0]);
    int to = std::stoi(mapping[1]);
    if (from == logical_device_id) {
      copy_to.push_back(to);
    }
    if (to == logical_device_id) {
      CHECK(!source_replica_id.has_value());
      source_replica_id = from;
    }
  }
  RendezvousKey rendezvous_key =
      GetRendezvousKey(run_options, device, {}, channel_id_present,
                       /*use_global_device_ids=*/std::nullopt, op_id);

  int rank = RankInGlobalDevices(rendezvous_key.global_devices, device).value();

  CollectivesInterface* collectives = GetCollectivesImpl(run_options);

  auto communicator =
      collectives->GetCommunicator(rendezvous_key.global_devices, rank).value();
  TF_CHECK_OK(communicator->CollectivePermute(
      rendezvous_key, byte_size, source_replica_id, copy_to, input_buffer,
      output_buffer, DefaultCollectiveTimeout()));
}
}  // namespace
}  // namespace runtime
}  // namespace cpu
}  // namespace xla

extern "C" {

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY int __xla_cpu_runtime_PrintfToStderr(
    const char* format, ...) {
  VLOG(3) << "__xla_cpu_runtime_PrintfToStderr " << format;
  va_list args;
  va_start(args, format);
  int result = vfprintf(stderr, format, args);
  va_end(args);
  return result;
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY int64_t __xla_cpu_runtime_TracingStart(
    const void* /* ExecutableRunOptions*  run_options_ptr*/, const char* name) {
  VLOG(3) << "TracingStart " << name;
  return tsl::profiler::TraceMe::ActivityStart(name);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_TracingEnd(
    const void* /* ExecutableRunOptions*  run_options_ptr*/, int64_t id) {
  VLOG(3) << "TracingEnd " << id;
  tsl::profiler::TraceMe::ActivityEnd(id);
}

void* __xla_cpu_runtime_AcquireInfeedBufferForDequeue(
    const xla::ExecutableRunOptions* run_options, int32_t buffer_length,
    const void* shape, int32_t shape_length) {
  return xla::cpu::runtime::AcquireInfeedBufferForDequeueImpl(
      run_options, buffer_length, shape, shape_length);
}

void __xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue(
    const xla::ExecutableRunOptions* run_options, int32_t buffer_length,
    void* buffer_ptr, const void* shape_ptr, int32_t shape_length) {
  return xla::cpu::runtime::ReleaseInfeedBufferAfterDequeueImpl(
      run_options, buffer_length, buffer_ptr, shape_ptr, shape_length);
}

void* __xla_cpu_runtime_AcquireOutfeedBufferForPopulation(
    const xla::ExecutableRunOptions* run_options, int32_t buffer_length,
    const void* shape_ptr, int32_t shape_length) {
  return xla::cpu::runtime::AcquireOutfeedBufferForPopulationImpl(
      run_options, buffer_length, shape_ptr, shape_length);
}

void __xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation(
    const xla::ExecutableRunOptions* run_options, int32_t buffer_length,
    void* buffer_ptr, const void* shape_ptr, int32_t shape_length) {
  return xla::cpu::runtime::ReleaseOutfeedBufferAfterPopulationImpl(
      run_options, buffer_length, buffer_ptr, shape_ptr, shape_length);
}

void __xla_cpu_runtime_AllToAll(const xla::ExecutableRunOptions* run_options,
                                int32_t channel_id_present, int64_t op_id,
                                const void* replica_groups_str,
                                int32_t replica_groups_str_size,
                                int32_t num_buffers, int64_t buffer_size,
                                void** source_buffers,
                                void** destination_buffers) {
  return xla::cpu::runtime::AllToAllImpl(
      run_options, channel_id_present, op_id, replica_groups_str,
      replica_groups_str_size, num_buffers, buffer_size, source_buffers,
      destination_buffers);
}

void __xla_cpu_runtime_AllGather(const xla::ExecutableRunOptions* run_options,
                                 int32_t channel_id_present,
                                 int32_t use_global_device_ids, int64_t op_id,
                                 const void* replica_groups_str,
                                 int32_t replica_groups_str_size,
                                 int64_t buffer_size, void* source_buffer,
                                 void* destination_buffer) {
  return xla::cpu::runtime::AllGatherImpl(
      run_options, channel_id_present, use_global_device_ids, op_id,
      replica_groups_str, replica_groups_str_size, buffer_size, source_buffer,
      destination_buffer);
}

void __xla_cpu_runtime_ReduceScatter(
    const xla::ExecutableRunOptions* run_options,
    const void* replica_groups_str, int32_t replica_groups_str_size,
    int32_t channel_id_present, int32_t use_global_device_ids, int64_t op_id,
    int32_t reduction_kind, int32_t element_type, int64_t chunk_elems,
    void* input_buffer, void* output_buffer) {
  return xla::cpu::runtime::ReduceScatterImpl(
      run_options, replica_groups_str, replica_groups_str_size,
      channel_id_present, use_global_device_ids, op_id, reduction_kind,
      element_type, chunk_elems, input_buffer, output_buffer);
}

void __xla_cpu_runtime_AllReduce(const xla::ExecutableRunOptions* run_options,
                                 const void* replica_groups_str,
                                 int32_t replica_groups_str_size,
                                 int32_t channel_id_present,
                                 int32_t use_global_device_ids, int64_t op_id,
                                 int32_t reduction_kind, const void* shape_ptr,
                                 int32_t shape_length, int32_t num_buffers,
                                 void** input_buffers, void** output_buffers) {

  std::cout << "__xla_cpu_runtime_AllReduce (cpu_runtime.cc)" << std::endl;
  return xla::cpu::runtime::AllReduceImpl(
      run_options, replica_groups_str, replica_groups_str_size,
      channel_id_present, use_global_device_ids, op_id, reduction_kind,
      shape_ptr, shape_length, num_buffers, input_buffers, output_buffers, /* is_async= */ false);
}

void __xla_cpu_runtime_AllReduceStart(const xla::ExecutableRunOptions* run_options,
                                 const void* replica_groups_str,
                                 int32_t replica_groups_str_size,
                                 int32_t channel_id_present,
                                 int32_t use_global_device_ids, int64_t op_id,
                                 int32_t reduction_kind, const void* shape_ptr,
                                 int32_t shape_length, int32_t num_buffers,
                                 void** input_buffers, void** output_buffers) {

  std::cout << "__xla_cpu_runtime_AllReduceStart (cpu_runtime.cc)  op_id="<< op_id << std::endl;
  return xla::cpu::runtime::AllReduceImpl(
      run_options, replica_groups_str, replica_groups_str_size,
      channel_id_present, use_global_device_ids, op_id, reduction_kind,
      shape_ptr, shape_length, num_buffers, input_buffers, output_buffers, /* is_async= */ true);
}

void __xla_cpu_runtime_AllReduceDone(const xla::ExecutableRunOptions* run_options,
                                 const void* replica_groups_str,
                                 int32_t replica_groups_str_size,
                                 int32_t channel_id_present,
                                 int32_t use_global_device_ids, int64_t op_id) {

  std::cout << "__xla_cpu_runtime_AllReduceDone (cpu_runtime.cc)" << std::endl;
  return;
  // return xla::cpu::runtime::DoneImpl(
  //     run_options, replica_groups_str, replica_groups_str_size,
  //     channel_id_present, use_global_device_ids, op_id);
}


void __xla_cpu_runtime_AsyncDone(const xla::ExecutableRunOptions* run_options,
                                 const void* replica_groups_str,
                                 int32_t replica_groups_str_size,
                                 int32_t channel_id_present,
                                 int32_t use_global_device_ids, int64_t op_id) {

  std::cout << "__xla_cpu_runtime_AsyncDone (cpu_runtime.cc)" << std::endl;
  return xla::cpu::runtime::DoneImpl(
      run_options, replica_groups_str, replica_groups_str_size,
      channel_id_present, use_global_device_ids, op_id);
}



void __xla_cpu_runtime_ReplicaId(const xla::ExecutableRunOptions* run_options,
                                 void* output_buffer) {
  return xla::cpu::runtime::ReplicaIdImpl(run_options, output_buffer);
}

void __xla_cpu_runtime_PartitionId(const xla::ExecutableRunOptions* run_options,
                                   void* output_buffer) {
  return xla::cpu::runtime::PartitionIdImpl(run_options, output_buffer);
}

void __xla_cpu_runtime_CollectivePermute(
    const xla::ExecutableRunOptions* run_options, int32_t channel_id_present,
    int64_t op_id, int32_t byte_size, void* input_buffer, void* output_buffer,
    const void* source_target_pairs, int32_t source_target_pairs_size) {
  return xla::cpu::runtime::CollectivePermuteImpl(
      run_options, channel_id_present, op_id, byte_size, input_buffer,
      output_buffer, source_target_pairs, source_target_pairs_size);
}

}  // extern "C"
