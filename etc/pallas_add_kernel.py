import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def add_kernel(x_ref, y_ref, o_ref):
    x = x_ref[:]
    y = y_ref[:]
    o_ref[:] = x + y


x, y = jnp.arange(8), jnp.arange(8, 16)
add = pl.pallas_call(add_kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.int32))
add(x, y)
