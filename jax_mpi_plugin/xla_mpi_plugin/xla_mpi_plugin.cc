#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tfrt_cpu_pjrt_client_mpi.h"

namespace my_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(pjrt::ActualStructSizeIsGreaterOrEqual("PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,args->struct_size));
  // TODO(b/263170683): cpu_device_count should be configurable after config options can be passed to PJRT_Client_Create.
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,xla::GetTfrtCpuClientMPI(/*asynchronous=*/true));
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_CpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{tsl::errors::Unimplemented(
      "Topology not supported for CPU compilation.")};
}

}  // namespace my_plugin

constexpr PJRT_Api pjrt_api = pjrt::CreatePjrtApi(my_plugin::PJRT_Client_Create, my_plugin::PJRT_CpuDeviceTopology_Create, pjrt::PJRT_Plugin_Initialize_NoOp);

extern "C" {
__attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi();
}



const PJRT_Api* GetPjrtApi() { return &pjrt_api; }


