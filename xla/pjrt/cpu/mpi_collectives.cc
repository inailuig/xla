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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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

absl::StatusOr<MPI_Datatype> PrimitiveTypeToMpiType(
    PrimitiveType element_type) {
  switch (element_type) {
    case S8:
      return MPI_INT8_T;
    case U8:
    case PRED:
      return MPI_UINT8_T;
    case S16:
      return MPI_INT16_T;
    case U16:
      return MPI_UINT16_T;
    case S32:
      return MPI_INT32_T;
    case U32:
      return MPI_UINT32_T;
    case S64:
      return MPI_INT64_T;
    case U64:
      return MPI_UINT64_T;
    case F32:
      return MPI_FLOAT;
    case F64:
      return MPI_DOUBLE;
    case C64:
      return MPI_C_COMPLEX;
    case C128:
      return MPI_C_DOUBLE_COMPLEX;
    default:
      // For implementing the reduction of unsupported types
      // see e.g. https://stackoverflow.com/a/29643391
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported primitive type for reduction: ",
          primitive_util::LowercasePrimitiveTypeName(element_type)));
  }
}

bool MpiTypeIsComplex(MPI_Datatype type) {
  return type == MPI_C_COMPLEX || type == MPI_C_DOUBLE_COMPLEX;
}

absl::StatusOr<MPI_Op> ReductionKindToMpiOp(ReductionKind reduction_kind,
                                            MPI_Datatype type) {
  switch (reduction_kind) {
    case ReductionKind::SUM:
      return MPI_SUM;
    case ReductionKind::PRODUCT:
      return MPI_PROD;
    case ReductionKind::MIN:
      if (!MpiTypeIsComplex(type)) {
        return MPI_MIN;
      } else {
        return absl::InvalidArgumentError(
            "MIN reduction not supported for complex types");
      }
    case ReductionKind::MAX:
      if (!MpiTypeIsComplex(type)) {
        return MPI_MAX;
      } else {
        return absl::InvalidArgumentError(
            "MAX reduction not supported for complex types");
      }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown reduction kind: ", reduction_kind));
  }
}

static absl::Status MpiErrorToAbslStatus(int error) {
  if (error != MPI_SUCCESS) {
    char error_str[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(error, error_str, &len);
    return absl::UnknownError(absl::StrCat("MPI error: ", error_str));
  }
  return absl::OkStatus();
}

MpiCollectivesCommunicator::MpiCollectivesCommunicator(
    std::vector<int> mpi_world_ranks) {
  MPI_Group group_world, group;

  // Create a MPI communicator conatining the ranks from mpi_world_ranks.
  // It's MPI ranks respect the order of mpi_world_ranks, which means that the
  // MPI communicator rank and MpiCollectivesCommunicator rank coincide.

  MPI_Comm_group(MPI_COMM_WORLD, &group_world);
  MPI_Group_incl(group_world, mpi_world_ranks.size(), mpi_world_ranks.data(),
                 &group);
  MPI_Comm_create(MPI_COMM_WORLD, group, &comm_);
  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);

  MPI_Group_free(&group_world);
  MPI_Group_free(&group);
}

MpiCollectivesCommunicator::~MpiCollectivesCommunicator() {
  MPI_Comm_free(&comm_);
};

absl::Status MpiCollectivesCommunicator::AllReduce(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t num_elements, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  TF_ASSIGN_OR_RETURN(MPI_Datatype type, PrimitiveTypeToMpiType(element_type));
  TF_ASSIGN_OR_RETURN(MPI_Op op, ReductionKindToMpiOp(reduction_kind, type));
  return MpiErrorToAbslStatus(MPI_Allreduce(input_buffer, output_buffer,
                                            num_elements, type, op, comm_));
}

absl::Status MpiCollectivesCommunicator::CollectivePermute(
    const RendezvousKey& key, size_t num_bytes, std::optional<int> source_rank,
    absl::Span<int const> target_ranks, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  int tag = 0;  // TODO come up with better tags.

  const int rank = mpi_rank_;

  std::vector<MPI_Request> requests;

  if (source_rank) {
    if (*source_rank == rank) {
      std::memcpy(output_buffer, input_buffer, num_bytes);
    } else {
      VLOG(1) << "recv at " << rank << " from " << *source_rank;
      requests.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Irecv(output_buffer, num_bytes, MPI_BYTE, *source_rank, tag,
                    comm_, &requests.back())));
    }
  } else {
    std::memset(output_buffer, 0, num_bytes);
  }

  for (int target : target_ranks) {
    if (target != rank) {
      VLOG(1) << "send from " << rank << " to " << target;
      requests.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Isend(input_buffer, num_bytes, MPI_BYTE, target, tag, comm_,
                    &requests.back())));
    }
  }

  for (auto& request : requests) {
    TF_RETURN_IF_ERROR(
        MpiErrorToAbslStatus(MPI_Wait(&request, MPI_STATUS_IGNORE)));
  }

  return absl::OkStatus();
}

absl::Status MpiCollectivesCommunicator::AllToAll(
    const RendezvousKey& key, size_t chunk_bytes,
    absl::Span<const void* const> input_buffers,
    absl::Span<void* const> output_buffers, absl::Duration timeout) {
  // We can't use MPI_Alltoall directly because it assumes that the inputs and
  // outputs are contiguous. Therefore here we implement it using MPI_Sendrecv.

  int tag = 0;  // TODO use better tags.
  const int rank = mpi_rank_;
  const int size = mpi_size_;
  TF_RET_CHECK(size == input_buffers.size());
  TF_RET_CHECK(size == output_buffers.size());

  std::memcpy(output_buffers[rank], input_buffers[rank], chunk_bytes);

  for (int i = 1; i < size; i++) {
    int send_rank = (rank + i) % size;
    int recv_rank = (rank + size - i) % size;
    TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
        MPI_Sendrecv(input_buffers[send_rank], chunk_bytes, MPI_BYTE, send_rank,
                     tag, output_buffers[recv_rank], chunk_bytes, MPI_BYTE,
                     recv_rank, tag, comm_, MPI_STATUS_IGNORE)));
  }

  return absl::OkStatus();
}

absl::Status MpiCollectivesCommunicator::AllGather(const RendezvousKey& key,
                                                   size_t chunk_bytes,
                                                   const void* input_buffer,
                                                   void* output_buffer,
                                                   absl::Duration timeout) {
  return MpiErrorToAbslStatus(MPI_Allgather(input_buffer, chunk_bytes, MPI_BYTE,
                                            output_buffer, chunk_bytes,
                                            MPI_BYTE, comm_));
}

absl::Status MpiCollectivesCommunicator::ReduceScatter(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t chunk_elems, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  const int size = mpi_size_;
  std::vector<int> recvcounts(size, chunk_elems);
  TF_ASSIGN_OR_RETURN(MPI_Datatype type, PrimitiveTypeToMpiType(element_type));
  TF_ASSIGN_OR_RETURN(MPI_Op op, ReductionKindToMpiOp(reduction_kind, type));
  return MpiErrorToAbslStatus(MPI_Reduce_scatter(
      input_buffer, output_buffer, recvcounts.data(), type, op, comm_));
}

MpiCollectives::MpiCollectives() {
  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size_);
  VLOG(1) << "MPI rank=" << mpi_world_rank_ << " size=" << mpi_world_size_;
}

MpiCollectives::~MpiCollectives() {
  contexts_.clear();
  MPI_Finalize();
}

absl::Status MpiCollectives::ExchangeGlobalDeviceIds(
    absl::Span<GlobalDeviceId const> global_devices, int rank) {
  // Build the mapping from global device id to mpi world rank by exchanging
  // messages between all ranks owning one of global_devices.
  // The argument rank here is the rank of this process inside global_devices
  // (not the mpi world rank).

  // set our own
  auto r = global_device_id_to_mpi_world_rank_.insert(
      std::make_pair(global_devices.at(rank), mpi_world_rank_));
  if (!r.second) {
    auto it = r.first;
    if (it->second != mpi_world_rank_) {
      return absl::UnknownError(absl::StrCat(
          "Inconsistent global device id ", global_devices.at(rank).value(),
          " on rank ", mpi_world_rank_, ". It should be on rank ", it->second,
          "."));
    }
  }

  // TODO come up with better tags.
  int tag_gid = 1;
  int tag_ack = 2;

  // cast to int64 for sending over MPI
  int64_t gid = static_cast<int64_t>(global_devices.at(rank).value());
  char dummy_ack_message;

  std::vector<MPI_Request> recv_requests_gid;
  // the following contains also a (unused) entry for this rank itself,
  // to facilitate indexing with the source/target rank
  std::vector<MPI_Request> recv_requests_ack(mpi_world_size_);
  std::vector<MPI_Request> send_requests_gid;
  std::vector<MPI_Request> send_requests_ack;

  std::vector<char> recv_buffer_ack(mpi_world_size_);
  std::vector<int64_t> recv_buffer_gid(mpi_world_size_);

  // Speculative receive requests for the global device id and ack message from
  // all mpi ranks
  for (int source = 0; source < mpi_world_size_; source++) {
    if (source != mpi_world_rank_) {
      recv_requests_gid.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Irecv(&recv_buffer_gid[source], 1, MPI_INT64_T, source, tag_gid,
                    MPI_COMM_WORLD, &recv_requests_gid.back())));
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Irecv(&recv_buffer_ack[source], 1, MPI_CHAR, source, tag_ack,
                    MPI_COMM_WORLD, &recv_requests_ack[source])));
    }
  }

  // Send the global_device_id of this mpi rank to all other ranks
  for (int target = 0; target < mpi_world_size_; target++) {
    if (target != mpi_world_rank_) {
      send_requests_gid.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Isend(&gid, 1, MPI_INT64_T, target, tag_gid, MPI_COMM_WORLD,
                    &send_requests_gid.back())));
    }
  }

  // Send ack to the participating mpi ranks the global device id is already
  // known of and assemble a set of global device ids for which the mpi rank is
  // not yet known.
  absl::flat_hash_set<GlobalDeviceId> unknown_global_device_ids;
  for (GlobalDeviceId id : global_devices) {
    auto it = global_device_id_to_mpi_world_rank_.find(id);
    if (it != global_device_id_to_mpi_world_rank_.end()) {
      int target = it->second;
      send_requests_ack.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Isend(&dummy_ack_message, 1, MPI_CHAR, target, tag_ack,
                    MPI_COMM_WORLD, &send_requests_ack.back())));
    } else {
      unknown_global_device_ids.insert(id);
    }
  }

  // Parse gid messages until the mpi rank of all unknown participating global
  // device ids has been received
  while (unknown_global_device_ids.size() > 0) {
    int indx;
    MPI_Status status;
    TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
        MPI_Waitany(recv_requests_gid.size(), recv_requests_gid.data(), &indx,
                    &status)));
    int rank_recv_from = status.MPI_SOURCE;

    recv_requests_gid.erase(recv_requests_gid.begin() + indx);

    GlobalDeviceId id(recv_buffer_gid.at(rank_recv_from));

    VLOG(1) << "MPI rank " << mpi_world_rank_ << " received global device id "
            << recv_buffer_gid.at(rank_recv_from) << " from rank "
            << rank_recv_from;

    auto it = unknown_global_device_ids.find(id);
    if (it != unknown_global_device_ids.end()) {
      unknown_global_device_ids.erase(it);
      // Send ack to the rank the previously unknown gid has been received
      // from
      send_requests_ack.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Isend(&dummy_ack_message, 1, MPI_CHAR, rank_recv_from, tag_ack,
                    MPI_COMM_WORLD, &send_requests_ack.back())));
      // Note that if there are several communicators (on disjunct sets of mpi
      // ranks) being requested at the same time, messages from mpi ranks not
      // participating in this one might arrive. The received rank<->gid mapping
      // in global_device_id_to_mpi_world_rank_ is still set for future use but
      // no ack is sent back.
      auto r = global_device_id_to_mpi_world_rank_.insert(
          std::make_pair(id, rank_recv_from));
      if (!r.second) {
        auto it = r.first;
        if (it->second != rank_recv_from) {
          return absl::UnknownError(
              absl::StrCat("MPI: rank ", mpi_world_rank_,
                           " received inconsistent global device id ",
                           id.value(), " from rank ", rank_recv_from,
                           ". It should be on rank ", it->second, "."));
        }
      }
    }
  }

  // Wait until all acks sent have been recieved
  TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(MPI_Waitall(
      send_requests_ack.size(), send_requests_ack.data(), MPI_STATUS_IGNORE)));

  // Wait until ack has been received from all involved ranks (except this
  // rank)
  for (GlobalDeviceId id : global_devices) {
    int target = global_device_id_to_mpi_world_rank_.at(id);
    if (target != mpi_world_rank_) {
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Wait(&recv_requests_ack.at(target), MPI_STATUS_IGNORE)));
    }
  }

  // Remove unused request for this rank itself
  recv_requests_ack.erase(recv_requests_ack.begin() + mpi_world_rank_);
  // and cancel all remaining requests
  for (auto requests :
       {send_requests_gid, recv_requests_ack, recv_requests_gid}) {
    for (auto& request : requests) {
      int flag;
      TF_RETURN_IF_ERROR(
          MpiErrorToAbslStatus(MPI_Test(&request, &flag, MPI_STATUS_IGNORE)));
      if (!flag) {
        TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(MPI_Cancel(&request)));
        TF_RETURN_IF_ERROR(
            MpiErrorToAbslStatus(MPI_Wait(&request, MPI_STATUS_IGNORE)));
      }
    }
  }
  return absl::OkStatus();
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

  TF_RETURN_IF_ERROR(ExchangeGlobalDeviceIds(global_devices, rank));

  std::vector<int> mpi_world_ranks;
  for (GlobalDeviceId id : global_devices) {
    auto it = global_device_id_to_mpi_world_rank_.find(id);
    if (it != global_device_id_to_mpi_world_rank_.end()) {
      mpi_world_ranks.push_back(it->second);
    } else {
      return absl::UnknownError(absl::StrCat(
          "MPI: Unknown mpi rank for GlobalDeviceId ", id.value()));
    }
  }
  context =
      std::make_shared<MpiCollectivesCommunicator>(std::move(mpi_world_ranks));
  return context;
}

}  // namespace xla::cpu
