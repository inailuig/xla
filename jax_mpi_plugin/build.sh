#! /bin/bash

python3 download_bazel.py
./bazel-6.1.2-linux-x86_64 build -c opt --nocheck_visibility xla_mpi_plugin:xla_mpi_plugin.so
python3 -m build ./python_package --outdir ./
