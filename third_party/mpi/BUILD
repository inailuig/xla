# from https://github.com/tensorflow/networking/pull/6/files#diff-39b035c7f0dee32bf072bcc1a58301fa412c55cc432780df590c3c7d04fb08e5
licenses(["restricted"])



cc_library(
    name = "mpi",
    srcs = ["libmpi.so"],
    hdrs = ["mpi.h", "mpi_portable_platform.h"], # ["mpi.h", "mpio.h", "mpicxx.h"]
    visibility = ["//visibility:public"],
    includes = ["include"],
)
