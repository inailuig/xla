import jax
from functools import partial
import jax.numpy as jnp
import numpy as np


jax.distributed.initialize()

print('localdev', jax.local_devices())
print('dev', jax.devices())

for d in jax.devices():
   print(d.id, d.host_id, d.process_index, d.task_id)

x = np.ones(4)+jax.process_index()

d = list(jax.devices())[-1]
print(jax.process_index(), 'chosen', d)
y = jax.device_put(x, d)
print('yd', y.device(), y.is_fully_addressable, y)
jax.distributed.shutdown()
