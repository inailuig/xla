__How to build on Ubuntu (tested with 22.04 LTS):__

```console
sudo apt update
sudo apt install -y git g++ python3-venv python3-virtualenv python3 python3-dev python3-numpy python3-wheel python3-build 
openmpi-bin libopenmpi-dev

git clone --branch mpi_allreduce_plugin https://github.com/inailuig/xla.git
cd xla/jax_mpi_plugin

./build.sh
```

Then just `pip install` the generated `jax_mpi-0.4.20-py3-none-manylinux2014_x86_64.whl`.
