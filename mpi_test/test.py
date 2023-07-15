import jax
from functools import partial
import jax.numpy as jnp

jax.distributed.initialize()

print('dev', jax.devices())
print('localdev', jax.local_devices())

@partial(jax.jit, out_shardings=jax.sharding.PositionalSharding(jax.devices()).reshape(-1,1,1))
def test(x):
    return x

@jax.jit
def s(x):
    return x.sum(axis=0)


x = jnp.ones((16, 5,3), dtype=jnp.float32)
x = test(x)
#print(s(x))

print(x.sharding)
