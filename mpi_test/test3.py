import jax
from functools import partial
import numpy as np
import jax.numpy as jnp

jax.distributed.initialize()

print('dev',jax.devices())
print('localdev', jax.local_devices())

#@partial(jax.jit, out_shardings=jax.sharding.PositionalSharding(jax.devices()).reshape(-1,1,1))
#def test(x):
#    return x

def put_global(local_array):
    local_shape = local_array.shape
    global_shape = (local_shape[0]*jax.process_count(),)+ local_shape[1:]
    sharding = jax.sharding.PositionalSharding(jax.devices()).reshape((-1,)+(1,)*(local_array.ndim-1))
    arrays = jax.device_put(jnp.split(local_array, jax.local_device_count(), axis = 0), jax.local_devices())
    return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)


@jax.jit
def s(x):
    return x.sum(axis=0)


x = np.ones((16, 5,3), dtype=jnp.float32)





#x = test(x)
#print(s(x))

x = put_global(x)

print(x.sharding)
