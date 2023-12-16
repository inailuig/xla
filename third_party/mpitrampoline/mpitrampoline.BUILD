# Description:
#  A forwarding MPI implementation that can use any other MPI implementation via an MPI ABI

load("@bazel_skylib//rules:expand_template.bzl", "expand_template")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE.md"])

genrule(
    name = "mpi_version",
    srcs = [
        "CMakeLists.txt",
        "include/mpi_version.h.in",
    ],
    outs = ["include/mpi_version.h"],
    cmd = """
      PROJECT_VERSION=`cat $(location CMakeLists.txt) \
                       | grep "MPItrampoline VERSION" | awk '{print $$NF}'`
      PROJECT_VERSION_MAJOR=`echo $$PROJECT_VERSION | cut -d. -f1`
      PROJECT_VERSION_MINOR=`echo $$PROJECT_VERSION | cut -d. -f2`
      PROJECT_VERSION_PATCH=`echo $$PROJECT_VERSION | cut -d. -f3`
      sed -e "s/@PROJECT_VERSION@/$${PROJECT_VERSION}/" \
          -e "s/@PROJECT_VERSION_MAJOR@/$${PROJECT_VERSION_MAJOR}/" \
          -e "s/@PROJECT_VERSION_MINOR@/$${PROJECT_VERSION_MINOR}/" \
          -e "s/@PROJECT_VERSION_PATCH@/$${PROJECT_VERSION_PATCH}/" \
          $(location include/mpi_version.h.in) > $(location include/mpi_version.h)
      """,
    )

expand_template(
    name = "mpi_defaults",
    out = "src/mpi_defaults.h",
    substitutions = {
        "@MPITRAMPOLINE_DEFAULT_DELAY_INIT@": "",
        "@MPITRAMPOLINE_DEFAULT_DLOPEN_BINDING@": "",
        "@MPITRAMPOLINE_DEFAULT_DLOPEN_MODE@": "",
        "@MPITRAMPOLINE_DEFAULT_LIB@": "",
        "@MPITRAMPOLINE_DEFAULT_PRELOAD@": "",
        "@MPITRAMPOLINE_DEFAULT_VERBOSE@": "",
    },
    template = "src/mpi_defaults.h.in",
)

genrule(
    name = "run_gen_decl",
    srcs = [
        "gen/gen_decl.py",
        "mpiabi/mpi_constants.py",
        "mpiabi/mpi_functions.py",
    ],
    outs = [
        "include/mpi_decl_constants_c.h",
        "include/mpi_decl_functions_c.h",
    ],
    cmd = "$(location gen/gen_decl.py) $(location include/mpi_decl_constants_c.h) \
           $(location include/mpi_decl_functions_c.h)",
)

genrule(
    name = "run_gen_defn",
    srcs = [
        "gen/gen_defn.py",
        "mpiabi/mpi_constants.py",
        "mpiabi/mpi_functions.py",
    ],
    outs = [
        "include/mpi_defn_constants_c.h",
        "include/mpi_defn_functions_c.h",
    ],
    cmd = "$(location gen/gen_defn.py) $(location include/mpi_defn_constants_c.h) \
           $(location include/mpi_defn_functions_c.h)",
)

genrule(
    name = "run_gen_init",
    srcs = [
        "gen/gen_init.py",
        "mpiabi/mpi_constants.py",
        "mpiabi/mpi_functions.py",
    ],
    outs = [
        "include/mpi_init_constants_c.h",
        "include/mpi_init_functions_c.h",
    ],
    cmd = "$(location gen/gen_init.py) $(location include/mpi_init_constants_c.h) \
           $(location include/mpi_init_functions_c.h)",
)

cc_library(
    name = "mpitrampoline",
    srcs = [
        "src/mpi.c",
    ],
    copts = [
        "-fexceptions",
    ],
    includes = ["include", "mpiabi", "src"],
    hdrs = [
        "include/mpi.h",
        "include/mpi_version.h",
        "include/mpi_decl_constants_c.h",
        "include/mpi_decl_functions_c.h",
        "include/mpi_defn_constants_c.h",
        "include/mpi_defn_functions_c.h",
        "include/mpi_init_constants_c.h",
        "include/mpi_init_functions_c.h",
        "mpiabi/mpiabi.h",
        "src/mpi_defaults.h",

    ]
)