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

@jax.jit
def s(x):
    #return jax.lax.add(x,x)
    return x.sum(axis=0)


x = np.ones((4, 3,5), dtype=jnp.complex64)

local_shape = x.shape
global_shape = (local_shape[0]*jax.process_count(),)+ local_shape[1:]
sharding = jax.sharding.PositionalSharding(jax.devices()).reshape((-1,)+(1,)*(x.ndim-1))
print('sss', sharding)
ldc = jax.local_device_count()
print('ldc', ldc)
aa = np.split(x, ldc, axis = 0)
print('aa', aa)
arrays = jax.device_put(aa, jax.local_devices())
print('aaa', arrays)
y = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)



#x = test(x)
#print(s(x))

#x = put_global(x)

print(y.sharding)

z = s(y)
print('zz', z.addressable_shards)

jax.distributed.shutdown()
