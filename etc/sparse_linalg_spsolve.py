import jax.numpy as jnp
from jax.experimental.sparse.linalg import spsolve

data = jnp.array([3.0, 2.0, 1.0, -1.0, 5.0, 1.0])
indices = jnp.array([0, 1, 0, 1, 1, 2], dtype=jnp.int32)
indptr = jnp.array([0, 2, 4, 6], dtype=jnp.int32)

b = jnp.array([2.0, -1.0, 2.0])
x = spsolve(data, indices, indptr, b)

print("Solution x:", x)
