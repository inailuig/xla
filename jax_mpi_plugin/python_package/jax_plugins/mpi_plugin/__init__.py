import os
import jax._src.xla_bridge as xb
def initialize():
  path = os.path.join(os.path.dirname(__file__), 'xla_mpi_plugin.so')
  xb.register_plugin('mpi', priority=500, library_path=path, options=None)
