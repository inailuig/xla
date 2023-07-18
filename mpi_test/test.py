from jax.config import config as jax_config
jax_config.update("jax_threefry_partitionable", True)
import jax
from functools import partial
import jax.numpy as jnp
from jax.sharding import PositionalSharding

print('local_devices', jax.local_devices())
print('devices', jax.devices())

for d in jax.devices():
   print(d.id, d.host_id, d.process_index, d.task_id)


@partial(jax.jit, out_shardings=PositionalSharding(jax.devices())) #.reshape(-1, 1))
def r(k):
    return jax.random.normal(k, (12,))

@jax.jit
def s(x):
    return x.sum(axis=0)

k = jax.random.PRNGKey(123)
y = r(k)
z = s(y)


jax.debug.visualize_array_sharding(y)
print(y.devices(), y.sharding.shape, y.is_fully_addressable, y.addressable_shards)
print(z.devices(), z.sharding.shape, z.is_fully_replicated, z.addressable_shards)
