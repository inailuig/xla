import jax
from functools import partial
import jax.numpy as jnp

jax.distributed.initialize()

print('localdev', jax.local_devices())
print('dev', jax.devices())


jax.distributed.shutdown()
