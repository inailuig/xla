/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/pjrt/cpu/mpi_collectives.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mpi.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/global_device_id.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla::cpu {

static absl::Status PrimitiveTypeToMpiType(PrimitiveType element_type,
                                           MPI_Datatype& type) {
  switch (element_type) {
    case S8:
      type = MPI_INT8_T;
      break;
    case U8:
    case PRED:
      type = MPI_UINT8_T;
      break;
    case S16:
      type = MPI_INT16_T;
      break;
    case U16:
      type = MPI_UINT16_T;
      break;
    case S32:
      type = MPI_INT32_T;
      break;
    case U32:
      type = MPI_UINT32_T;
      break;
    case S64:
      type = MPI_INT64_T;
      break;
    case U64:
      type = MPI_UINT64_T;
      break;
    case F32:
      type = MPI_FLOAT;
      break;
    case F64:
      type = MPI_DOUBLE;
      break;
    case C64:
      type = MPI_C_COMPLEX;
      break;
    case C128:
      type = MPI_C_DOUBLE_COMPLEX;
      break;
    default:
      // TODO implement reduction for the unsupported types
      // see e.g. https://stackoverflow.com/a/29643391
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported primitive type for reduction: ",
          primitive_util::LowercasePrimitiveTypeName(element_type)));
  }
  return absl::OkStatus();
}

bool MpiTypeIsComplex(MPI_Datatype type) {
  return type == MPI_C_COMPLEX || type == MPI_C_DOUBLE_COMPLEX;
}

static absl::Status ReductionKindToMpiOp(ReductionKind reduction_kind,
                                         MPI_Datatype type, MPI_Op& op) {
  switch (reduction_kind) {
    case ReductionKind::SUM:
      op = MPI_SUM;
      break;
    case ReductionKind::PRODUCT:
      op = MPI_PROD;
      break;
    case ReductionKind::MIN:
      // TODO implement custom complex max/min reduction
      if (!MpiTypeIsComplex(type)) {
        op = MPI_MIN;
      } else {
        return absl::InvalidArgumentError(
            "MIN reduction not supported for complex types");
      }
      break;
    case ReductionKind::MAX:
      // TODO implement custom complex max/min reduction
      if (!MpiTypeIsComplex(type)) {
        op = MPI_MAX;
      } else {
        return absl::InvalidArgumentError(
            "MAX reduction not supported for complex types");
      }
      break;
  }
  return absl::OkStatus();
}

static absl::Status mpi_error_to_absl_status(int error) {
  if (error != MPI_SUCCESS) {
    char error_str[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(error, error_str, &len);
    return absl::UnknownError(absl::StrCat("MPI error: ", error_str));
  }
  return absl::OkStatus();
}

MpiCollectivesCommunicator::MpiCollectivesCommunicator(
    std::vector<int> global_ranks) {
  MPI_Group group_world, group;

  MPI_Comm_group(MPI_COMM_WORLD, &group_world);
  MPI_Group_incl(group_world, global_ranks.size(), global_ranks.data(), &group);
  MPI_Comm_create(MPI_COMM_WORLD, group, &comm);

  MPI_Group_free(&group_world);
  MPI_Group_free(&group);
}

MpiCollectivesCommunicator::~MpiCollectivesCommunicator() {
  MPI_Comm_free(&comm);
};

absl::Status MpiCollectivesCommunicator::AllReduce(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t num_elements, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  MPI_Op op;
  MPI_Datatype type;
  TF_RETURN_IF_ERROR(PrimitiveTypeToMpiType(element_type, type));
  TF_RETURN_IF_ERROR(ReductionKindToMpiOp(reduction_kind, type, op));
  return mpi_error_to_absl_status(
      MPI_Allreduce(input_buffer, output_buffer, num_elements, type, op, comm));
}

absl::Status MpiCollectivesCommunicator::CollectivePermute(
    const RendezvousKey& key, size_t num_bytes, std::optional<int> source_rank,
    absl::Span<int const> target_ranks, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  int tag = 0;  // TODO come up with better tags.
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::vector<MPI_Request> requests;

  if (source_rank) {
    if (*source_rank == rank) {
      std::memcpy(output_buffer, input_buffer, num_bytes);
    } else {
      VLOG(1) << "recv at " << rank << " from " << *source_rank;
      requests.emplace_back();
      TF_RETURN_IF_ERROR(mpi_error_to_absl_status(
          MPI_Irecv(output_buffer, num_bytes, MPI_BYTE, *source_rank, tag, comm,
                    &requests.back())));
    }
  } else {
    std::memset(output_buffer, 0, num_bytes);
  }

  for (int target : target_ranks) {
    if (target != rank) {
      VLOG(1) << "send from " << rank << " to " << target;
      requests.emplace_back();
      TF_RETURN_IF_ERROR(mpi_error_to_absl_status(
          MPI_Isend(input_buffer, num_bytes, MPI_BYTE, target, tag, comm,
                    &requests.back())));
    }
  }

  for (auto& request : requests) {
    TF_RETURN_IF_ERROR(
        mpi_error_to_absl_status(MPI_Wait(&request, MPI_STATUS_IGNORE)));
  }

  return absl::OkStatus();
}

absl::Status MpiCollectivesCommunicator::AllToAll(
    const RendezvousKey& key, size_t chunk_bytes,
    absl::Span<const void* const> input_buffers,
    absl::Span<void* const> output_buffers, absl::Duration timeout) {
  int tag = 0;  // TODO use better tags.
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  TF_RET_CHECK(size == input_buffers.size());
  TF_RET_CHECK(size == output_buffers.size());

  std::memcpy(output_buffers[rank], input_buffers[rank], chunk_bytes);

  for (int i = 1; i < size; i++) {
    int send_rank = (rank + i) % size;
    int recv_rank = (rank + size - i) % size;
    TF_RETURN_IF_ERROR(mpi_error_to_absl_status(
        MPI_Sendrecv(input_buffers[send_rank], chunk_bytes, MPI_BYTE, send_rank,
                     tag, output_buffers[recv_rank], chunk_bytes, MPI_BYTE,
                     recv_rank, tag, comm, MPI_STATUS_IGNORE)));
  }

  return absl::OkStatus();
}

absl::Status MpiCollectivesCommunicator::AllGather(const RendezvousKey& key,
                                                   size_t chunk_bytes,
                                                   const void* input_buffer,
                                                   void* output_buffer,
                                                   absl::Duration timeout) {
  return mpi_error_to_absl_status(MPI_Allgather(input_buffer, chunk_bytes,
                                                MPI_BYTE, output_buffer,
                                                chunk_bytes, MPI_BYTE, comm));
}

absl::Status MpiCollectivesCommunicator::ReduceScatter(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t chunk_elems, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  int size;
  MPI_Comm_size(comm, &size);
  std::vector<int> recvcounts(size, chunk_elems);
  MPI_Op op;
  MPI_Datatype type;
  TF_RETURN_IF_ERROR(PrimitiveTypeToMpiType(element_type, type));
  TF_RETURN_IF_ERROR(ReductionKindToMpiOp(reduction_kind, type, op));
  return mpi_error_to_absl_status(MPI_Reduce_scatter(
      input_buffer, output_buffer, recvcounts.data(), type, op, comm));
}

MpiCollectives::MpiCollectives() {
  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  VLOG(1) << "MPI rank=" << mpi_rank << " size=" << mpi_size;
}

MpiCollectives::~MpiCollectives() {
  contexts_.clear();
  MPI_Finalize();
}

absl::StatusOr<std::shared_ptr<CollectivesCommunicator>>
MpiCollectives::GetCommunicator(absl::Span<GlobalDeviceId const> global_devices,
                                int rank) {
  int flag;
  MPI_Is_thread_main(&flag);
  if (!flag) {
    return absl::UnknownError(
        absl::StrCat("MPI: Communicator requested from a thread that is not "
                     "the one MPI was initialized from. Multiple "
                     "threads/devices per process are not yet supported."));
  }

  auto& context = contexts_[std::make_tuple(
      std::vector<GlobalDeviceId>(global_devices.begin(), global_devices.end()),
      rank)];
  if (context) {
    return context;
  }

  // we and that there is only one device per mpi rank
  // and that the mpi rank and global device id are the same,
  std::vector<int> global_ranks(global_devices.size());
  std::transform(global_devices.begin(), global_devices.end(),
                 global_ranks.begin(),
                 [](GlobalDeviceId i) { return static_cast<int>(i.value()); });
  context =
      std::make_shared<MpiCollectivesCommunicator>(std::move(global_ranks));
  return context;
}

}  // namespace xla::cpu
