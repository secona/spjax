import jax

from spjax.tensor import SparseTensor


def sparse_add(a: SparseTensor, b: SparseTensor) -> jax.Array:
    return a.add(b)


def sparse_dot(a: SparseTensor, b: SparseTensor) -> jax.Array:
    return a.dot(b)
