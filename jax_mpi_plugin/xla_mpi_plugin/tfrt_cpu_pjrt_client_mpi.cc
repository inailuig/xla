/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tfrt_cpu_pjrt_client_mpi.h"

#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/utils.h"
#include "third_party/mpi/mpi.h"


namespace xla {

Status BuildDistributedDevicesMPI(
    int mpi_rank, // mpi rank
    int mpi_size, // mpi size
    int cpu_device_count,
    int max_inflight_computations_per_device,
    std::vector<std::unique_ptr<TfrtCpuDevice>>* devices,
    std::map<int, GlobalDeviceId>* cpu_device_ids
  ) {

  std::vector<int> n_devices(mpi_size);
  // get how many threads (i.e. local cpu devices) each rank has
  MPI_Allgather(&cpu_device_count, 1, MPI_INT, n_devices.data(), 1, MPI_INT, MPI_COMM_WORLD);

  GlobalTopologyProto global_topology;

  int next_global_device_id = 0;
  for (int node_id=0; node_id<mpi_size; ++node_id){
      LocalTopologyProto* local_topology = global_topology.add_nodes();
      local_topology->set_node_id(node_id);
      for (int id=0; id < n_devices.at(node_id); ++id) {
          const int global_device_id = next_global_device_id++;
          DeviceProto* device_proto = local_topology->add_devices();
          device_proto->set_local_device_ordinal(id);
          device_proto->set_global_device_id(global_device_id);

          devices->push_back(std::make_unique<TfrtCpuDevice>(node_id, global_device_id, id,  max_inflight_computations_per_device));
          if (node_id == mpi_rank) {
            cpu_device_ids->insert(std::make_pair(id, global_device_id));
          }
      }
  }
  return OkStatus();
}


StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClientMPI(bool asynchronous) {
  return GetTfrtCpuClientMPI(asynchronous, CpuDeviceCount());
}

StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClientMPI(bool asynchronous,  int cpu_device_count, int max_inflight_computations_per_device) {
  VLOG(1) << "GetTfrtCpuClientMPI";

  // Need at least CpuDeviceCount threads to launch one collective.
  size_t num_threads = std::max(DefaultThreadPoolSize(), cpu_device_count);


  VLOG(1) << "calling MPI_Init";
  MPI_Init(NULL, NULL);
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  VLOG(1) << "MPI rank=" << mpi_rank << " size=" << mpi_size;



  std::vector<std::unique_ptr<TfrtCpuDevice>> devices;
  std::map<int, GlobalDeviceId> cpu_global_device_ids;
  TF_RETURN_IF_ERROR(BuildDistributedDevicesMPI(mpi_rank, mpi_size, cpu_device_count, max_inflight_computations_per_device, &devices, &cpu_global_device_ids));
  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtCpuClient>(mpi_rank, std::move(devices), num_threads, std::move(cpu_global_device_ids)));
}

}  // namespace xla
