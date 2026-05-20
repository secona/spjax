import jax
import jax.numpy as jnp
from jax.experimental import sparse

def spmv(M, v):
    return M @ v

batched_spmv = jax.vmap(sparse.sparsify(spmv), in_axes=(0, 0))

dense_batch = jnp.array([
    [[1., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]], # Mat 1
    [[0., 0., 0., 0.], [0., 0., 3., 0.], [0., 0., 0., 4.], [0., 0., 0., 0.]], # Mat 2
    [[5., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 6.]], # Mat 3
    [[0., 0., 0., 0.], [0., 7., 0., 0.], [0., 0., 8., 0.], [0., 0., 0., 0.]], # Mat 4
    [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [9., 0., 0., 10.]] # Mat 5
])

bcoo_batch = sparse.BCSR.fromdense(dense_batch, n_batch=1)

dense_vectors = jnp.ones((5, 4)) # 5 vectors of [1, 1, 1, 1]

outputs = batched_spmv(bcoo_batch, dense_vectors)

print("Output shape:", outputs.shape)
print("Result:\n", outputs)
