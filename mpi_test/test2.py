import jax
from functools import partial
import jax.numpy as jnp

jax.distributed.initialize()

print(jax.devices())
