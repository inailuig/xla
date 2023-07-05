import jax
from functools import partial
import jax.numpy as jnp

print(jax.devices())

@partial(jax.jit, out_shardings=jax.sharding.PositionalSharding(jax.devices()))
def test(x):
    return x
@jax.jit
def s(x, y):
    return x.sum(axis=-1), y.sum(axis=-1)

x = test(jnp.ones(32))
print(s(x, x))

