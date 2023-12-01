__How to build:__
tested on Ubuntu 22.04 LTS

```console
sudo apt update
sudo apt install -y git g++ python3-venv python3-virtualenv python3 python3-dev python3-numpy python3-wheel python3-build openmpi-bin libopenmpi-dev

git clone --branch mpi_allreduce_plugin https://github.com/inailuig/xla.git
cd xla/jax_mpi_plugin

./build.sh
```

Then just `pip install` the generated `jax_mpi-0.4.20-py3-none-manylinux2014_x86_64.whl`.


__How to use__

```python
from jax.config import config as jax_config
jax_config.update("jax_platforms", "mpi")

# optional but highly recommended
jax_config.update("jax_threefry_partitionable", True)
```

For hybrid paralelism (mpi & multiple threads/local cpu devices), reusing `OMP_NUM_THREADS` for our purpose set the following:
```python
import os
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={os.environ.get("OMP_NUM_THREADS")}'
```


