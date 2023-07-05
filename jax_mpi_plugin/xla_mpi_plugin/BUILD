load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")


cc_library(
    name = "tfrt_cpu_pjrt_client_mpi",
    srcs = ["tfrt_cpu_pjrt_client_mpi.cc"],
    hdrs = ["tfrt_cpu_pjrt_client_mpi.h"],
    visibility = [
        "//xla:friends",
    ],
    deps = [
        "//xla/pjrt:tfrt_cpu_pjrt_client",
        "//xla/pjrt:pjrt_client",
        "//third_party/mpi",
    ],
)


cc_binary(
    name = "xla_mpi_plugin.so",
    linkshared = 1,
    linkstatic = 1,
    linkopts = [
            "-Wl,--no-undefined",
    ],
    srcs = [
        "xla_mpi_plugin.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/pjrt:pjrt_client",
        ":tfrt_cpu_pjrt_client_mpi",
        "//xla/python/ifrt",
        "//xla/python/pjrt_ifrt",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//third_party/mpi",
    ],
)
