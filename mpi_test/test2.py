import jax
from functools import partial
import jax.numpy as jnp

jax.distributed.initialize()

print('localdev', jax.local_devices())
print('dev', jax.devices())

for d in jax.devices():
   print(d.id, d.host_id, d.process_index, d.task_id)

jax.distributed.shutdown()
