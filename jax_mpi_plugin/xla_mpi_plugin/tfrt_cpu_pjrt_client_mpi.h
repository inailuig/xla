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

#ifndef XLA_PJRT_TFRT_CPU_PJRT_CLIENT_MPI_H_
#define XLA_PJRT_TFRT_CPU_PJRT_CLIENT_MPI_H_

#include "xla/pjrt/pjrt_client.h"
#include "xla/statusor.h"
#include "third_party/mpi/mpi.h"

namespace xla {
 StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClientMPI(bool asynchronous);
 StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClientMPI(bool asynchronous,  int cpu_device_count, int max_inflight_computations_per_device = 32);
}  // namespace xla

#endif  // XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_
