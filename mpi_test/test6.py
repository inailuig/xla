from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update("jax_threefry_partitionable", True)


import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax.sharding import PositionalSharding

jax.distributed.initialize()

print('localdev', jax.local_devices())
print('dev', jax.devices())

for d in jax.devices():
   print(d.id, d.host_id, d.process_index, d.task_id)

k = jax.random.PRNGKey(123)

@partial(jax.jit, out_shardings=PositionalSharding(jax.devices())) #.reshape(-1, 1))
def test(k):
    #return jnp.arange(12)
    return jax.random.normal(k, (12,))



y = test(k)


jax.debug.visualize_array_sharding(y)
print('yd', y.devices(), y.sharding.shape, y.is_fully_addressable, y.addressable_shards)
jax.distributed.shutdown()
