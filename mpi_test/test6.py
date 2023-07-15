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

x = np.ones((12, 5))*jax.process_index()

@partial(jax.jit, out_shardings=PositionalSharding(jax.devices()).reshape(-1, 1))
def test(x):
    return x

@jax.jit
def m(x):
    return x.sum(axis=0) / x.mean(axis=0)

y = test(x)

y = m(y)

print('yd', y.devices(), y.sharding.shape, y.is_fully_addressable, y.addressable_shards[0])
jax.distributed.shutdown()
